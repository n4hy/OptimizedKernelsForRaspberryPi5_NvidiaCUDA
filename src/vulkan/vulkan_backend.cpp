/**
 * OptMathKernels Vulkan Compute Backend
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Cross-vendor GPU compute backend built on the Vulkan 1.2 compute pipeline.
 * Provides Eigen-typed host wrappers around SPIR-V compute shaders for vector,
 * matrix, DSP, reduction, scan, and FFT operations. Runs on any Vulkan 1.1+
 * device (NVIDIA, AMD, Intel, ARM Mali, llvmpipe), with a transparent CPU
 * fallback compiled in when OPTMATH_USE_VULKAN is undefined.
 *
 * ---------------------------------------------------------------------------
 * 1. SPIR-V Shader Loading  (readFile)
 * ---------------------------------------------------------------------------
 *    Locates compiled `.spv` modules by probing, in priority order: the
 *    OPTMATH_KERNELS_PATH env var, the compile-time OPTMATH_SPV_BUILD_DIR
 *    (for FetchContent consumers), the build tree, and the system install
 *    prefixes (/usr/local/share, /usr/share). Pads each blob to 4-byte
 *    alignment as required by vkCreateShaderModule.
 *
 * ---------------------------------------------------------------------------
 * 2. Device Memory & Buffers  (createBuffer, findMemoryType)
 * ---------------------------------------------------------------------------
 *    Allocates VkBuffers and backing VkDeviceMemory, selecting a memory type
 *    that satisfies the requested VkMemoryPropertyFlags (typically HOST_VISIBLE
 *    | HOST_COHERENT for staging host<->device data).
 *
 * ---------------------------------------------------------------------------
 * 3. Context Lifecycle  (VulkanContext::init / cleanup, atexit handler)
 * ---------------------------------------------------------------------------
 *    Creates the instance, selects a physical device, creates the logical
 *    device + compute queue + command pool, and tears them down in reverse.
 *    Physical-device selection ranks candidates by type and PREFERS A DISCRETE
 *    GPU (discrete > integrated > virtual > CPU), considering only devices that
 *    expose a VK_QUEUE_COMPUTE_BIT queue family. This avoids defaulting to a
 *    slower integrated GPU on multi-GPU machines (see the detailed comment at
 *    the selection site). An atexit handler runs cleanup before Vulkan layer
 *    static destructors to avoid shutdown crashes.
 *
 * ---------------------------------------------------------------------------
 * 4. Pipeline Cache  (getOrCreatePipeline, cleanupPipelineCache)
 * ---------------------------------------------------------------------------
 *    Builds and memoizes per-shader compute pipelines (descriptor set layout,
 *    pipeline layout with push constants, VkPipeline) in a mutex-guarded map
 *    keyed by shader name + buffer count + push-constant size, so repeated
 *    dispatches of the same kernel reuse GPU objects.
 *
 * ---------------------------------------------------------------------------
 * 5. Generic Compute Dispatch  (run_compute, run_reduction)
 * ---------------------------------------------------------------------------
 *    Stages input buffers, binds descriptor sets, records a command buffer,
 *    submits to the compute queue, and waits on a fence before reading back
 *    results. run_reduction implements multi-pass tree reduction for scalar
 *    outputs with a host-side combine of the final partials.
 *
 * ---------------------------------------------------------------------------
 * 6. Vector Operations
 * ---------------------------------------------------------------------------
 *    vulkan_vec_add / _sub / _mul / _div (elementwise), vulkan_vec_dot and
 *    vulkan_vec_norm (reduction-backed).
 *
 * ---------------------------------------------------------------------------
 * 7. Matrix Operations
 * ---------------------------------------------------------------------------
 *    Elementwise add/sub/scale/hadamard, vulkan_mat_mul (16x16 shared-memory
 *    tiled GEMM on the default/VideoCore path, 32x32 on Mali-G720),
 *    vulkan_mat_transpose, mat-vec product and outer product. Eigen matrices
 *    are column-major; shaders index accordingly.
 *
 * ---------------------------------------------------------------------------
 * 8. DSP Kernels
 * ---------------------------------------------------------------------------
 *    vulkan_convolution_1d/2d and vulkan_correlation_1d/2d.
 *
 * ---------------------------------------------------------------------------
 * 9. Reductions & Scan
 * ---------------------------------------------------------------------------
 *    vulkan_reduce_sum/max/min (via run_reduction) and vulkan_scan_prefix_sum
 *    (work-efficient Blelloch scan; falls back to CPU above the shader's
 *    supported workgroup size of 256).
 *
 * ---------------------------------------------------------------------------
 * 10. FFT  (vulkan_fft_radix2/radix4, bit_reverse_copy)
 * ---------------------------------------------------------------------------
 *    In-place iterative Cooley-Tukey FFT with bit-reversal permutation.
 *
 * ---------------------------------------------------------------------------
 * 11. Mali-G720 Specialization
 * ---------------------------------------------------------------------------
 *    Detects ARM Mali (vendor ID 0x13B5) and specifically Mali-G720-Immortalis
 *    to enable subgroup-optimized shader variants, with transparent fallback
 *    to standard shaders on all other GPUs.
 *
 * ---------------------------------------------------------------------------
 * 12. CPU Fallback Stub
 * ---------------------------------------------------------------------------
 *    When OPTMATH_USE_VULKAN is not defined, VulkanContext::init() returns
 *    false and is_available() reports unavailable, so callers transparently
 *    use the NEON/SVE2/Eigen paths instead.
 */
#include "optmath/vulkan_backend.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>

namespace optmath {
namespace vulkan {

// Only compile implementation if Vulkan is enabled
#ifdef OPTMATH_USE_VULKAN

// Utility: Read file
static std::vector<char> readFile(const std::string& filename) {
    // Try multiple paths:
    // 1. Environment variable OPTMATH_KERNELS_PATH
    // 2. Current directory
    // 3. Relative to executable (simplistic check for build dir structure)
    // 4. System install path (e.g. /usr/local/share/optmathkernels/shaders/)

    std::vector<std::string> paths;

    // Check environment variable
    if (const char* env_p = std::getenv("OPTMATH_KERNELS_PATH")) {
        std::string envPath = env_p;
        if (!envPath.empty() && envPath.back() != '/') envPath += '/';
        paths.push_back(envPath + filename);
    }

    // Compile-time SPV directory (works for FetchContent consumers)
#ifdef OPTMATH_SPV_BUILD_DIR
    paths.push_back(std::string(OPTMATH_SPV_BUILD_DIR) + "/" + filename);
#endif

    paths.push_back(filename);
    // Relative path for finding shaders in build tree when running from build root or examples
    paths.push_back("src/" + filename);
    paths.push_back("build/src/" + filename);
    paths.push_back("../src/" + filename);

    // Install paths
    paths.push_back("/usr/local/share/optmathkernels/shaders/" + filename);
    paths.push_back("/usr/share/optmathkernels/shaders/" + filename);

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            std::streampos pos = file.tellg();
            if (pos < 0) throw std::runtime_error("Failed to determine file size: " + filename);
            size_t fileSize = static_cast<size_t>(pos);
            // Pad to uint32_t alignment for Vulkan SPIR-V requirement
            size_t paddedSize = (fileSize + 3) & ~size_t(3);
            std::vector<char> buffer(paddedSize, 0);
            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();
            return buffer;
        }
    }
    throw std::runtime_error("failed to open file: " + filename);
}

// Utility: Create Buffer
static void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VulkanContext& ctx = VulkanContext::get();
    VkDevice device = ctx.device;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = ctx.findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        bufferMemory = VK_NULL_HANDLE;
        throw std::runtime_error("failed to bind buffer memory!");
    }
}

// -----------------------------------------------------------------------------
// VulkanContext
// -----------------------------------------------------------------------------

static void vulkanAtexitCleanup() {
    VulkanContext::get().cleanup();
}

VulkanContext& VulkanContext::get() {
    static VulkanContext instance;
    return instance;
}

bool VulkanContext::init() {
    if (initialized) return true;

    // Clean up any partially-created state from a previous failed init()
    if (device || instance) {
        cleanup();
    }

    // 1. Create Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "OptMathKernels";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // For simplicity, no validation layers in this minimal example
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create instance\n";
        return false;
    }

    // 2. Pick Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[Vulkan] Failed to find GPUs with Vulkan support\n";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // --- Physical-device selection: prefer a discrete GPU -----------------
    //
    // Vulkan does NOT guarantee any ordering of vkEnumeratePhysicalDevices, and
    // in practice the loader often lists the integrated GPU first on laptops
    // and workstations that also have a discrete card. The previous behaviour
    // here ("physicalDevice = devices[0]") therefore silently bound the slower
    // integrated GPU. For example, on an Intel Core Ultra 9 275HX + RTX 5070 Ti
    // box the loader enumerates: [0] Intel Graphics (ARL, integrated),
    // [1] NVIDIA RTX 5070 Ti (discrete), [2] llvmpipe (CPU) -- so devices[0]
    // was the iGPU.
    //
    // We instead score every candidate by its VkPhysicalDeviceType and keep the
    // highest-scoring one, giving the natural performance preference order:
    //   discrete (4) > integrated (3) > virtual (2) > CPU/llvmpipe (1) > other.
    //
    // Only candidates that advertise a VK_QUEUE_COMPUTE_BIT queue family are
    // eligible. This is what makes the separate compute-queue search further
    // below guaranteed to succeed for the device we pick here (a device could
    // otherwise expose only graphics/transfer queues, e.g. a display-only or
    // headless adapter).
    //
    // To override the automatic choice -- e.g. to benchmark the integrated GPU
    // -- restrict which ICD the loader sees, which limits enumeration to that
    // vendor's devices:
    //   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json   ./app   # iGPU
    //   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json  ./app   # NV
    //
    // deviceScore: maps a Vulkan device type to a preference rank (higher wins).
    auto deviceScore = [](VkPhysicalDeviceType type) -> int {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4; // dedicated card, fastest
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3; // shares system RAM
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2; // virtualized passthrough
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1; // software rasterizer (llvmpipe)
            default:                                     return 0; // VK_PHYSICAL_DEVICE_TYPE_OTHER
        }
    };
    // hasComputeQueue: true iff the device exposes at least one queue family
    // with the compute capability bit set (required to dispatch our shaders).
    auto hasComputeQueue = [](VkPhysicalDevice dev) -> bool {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qFamilies(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qFamilies.data());
        for (const auto& q : qFamilies) {
            if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) return true;
        }
        return false;
    };

    // Single pass: keep the compute-capable device with the highest type score.
    physicalDevice = VK_NULL_HANDLE;
    int bestScore = -1;
    for (const auto& candidate : devices) {
        if (!hasComputeQueue(candidate)) continue;   // skip non-compute adapters
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(candidate, &props);
        int score = deviceScore(props.deviceType);
        if (score > bestScore) {
            bestScore = score;
            physicalDevice = candidate;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        // No enumerated device advertised a compute queue. Fall back to the
        // first device so the compute-queue search below produces a single,
        // clear failure path rather than us erroring out here.
        physicalDevice = devices[0];
    } else {
        VkPhysicalDeviceProperties selProps;
        vkGetPhysicalDeviceProperties(physicalDevice, &selProps);
        std::cerr << "[Vulkan] Selected GPU: " << selProps.deviceName << "\n";
    }

    // 3. Find Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    bool found = false;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            found = true;
            break;
        }
        i++;
    }

    if (!found) {
        std::cerr << "[Vulkan] Failed to find a compute queue family\n";
        return false;
    }

    // 4. Create Logical Device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
         std::cerr << "[Vulkan] Failed to create logical device\n";
         return false;
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create command pool\n";
        return false;
    }

    // Detect GPU vendor and name for optimized shader selection
    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &devProps);
    std::string gpuName(devProps.deviceName);

    // Mali GPU detection (vendor ID 0x13B5 = ARM)
    isMaliGpu = (devProps.vendorID == 0x13B5);
    isMaliG720 = isMaliGpu && (gpuName.find("Mali-G720") != std::string::npos ||
                                gpuName.find("G720") != std::string::npos);

    // Broadcom VideoCore VII (Raspberry Pi 5) detection (vendor ID 0x14E4).
    // Exposed via the Mesa v3dv driver: subgroup width 16,
    // maxComputeWorkGroupInvocations = 256 (so the 1024-invocation Mali
    // variants must NOT be selected here). Uses the default tiled kernels.
    isBroadcomGpu = (devProps.vendorID == 0x14E4);

    if (isMaliGpu) {
        std::cerr << "[Vulkan] Mali GPU detected: " << gpuName << "\n";
        if (isMaliG720) {
            std::cerr << "[Vulkan] Mali-G720 Immortalis optimizations enabled\n";
        }
    } else if (isBroadcomGpu) {
        std::cerr << "[Vulkan] Broadcom VideoCore GPU detected: " << gpuName
                  << " (tiled compute kernels enabled)\n";
    }

    // Subgroup capability for the shuffle-based sum reduction. V3D does not
    // expose ARITHMETIC subgroup ops (subgroupAdd), but it does expose SHUFFLE,
    // from which we build the reduction. Requires: SHUFFLE + BASIC support in
    // compute, a power-of-two subgroup size, and numSubgroups (256/size) <=
    // size so the per-subgroup partials fit a single final subgroup reduction
    // (i.e. size >= 16 for the 256-wide workgroup).
    {
        VkPhysicalDeviceSubgroupProperties sgProps{};
        sgProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &sgProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        subgroupSize = sgProps.subgroupSize;
        const bool hasShuffle = (sgProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
        const bool hasBasic   = (sgProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
        const bool inCompute  = (sgProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        const bool pow2       = subgroupSize >= 16 && (subgroupSize & (subgroupSize - 1)) == 0;
        subgroupCanReduce = hasShuffle && hasBasic && inCompute && pow2;
        if (subgroupCanReduce) {
            std::cerr << "[Vulkan] Subgroup-shuffle reduction available (subgroupSize="
                      << subgroupSize << ")\n";
        }
    }

    initialized = true;

    // Register atexit handler so cleanup runs before static destructors of
    // Vulkan layers (e.g. libVkLayer_window_system_integration.so) which
    // would otherwise crash when their internal maps are already destroyed.
    std::atexit(vulkanAtexitCleanup);

    return true;
}

// Forward declaration for cleanup
static void cleanupPipelineCache(VkDevice device);
static void cleanupBufferPool(VkDevice device);

void VulkanContext::cleanup() {
    if (device) {
        vkDeviceWaitIdle(device);
        // Free pooled scratch buffers and the pipeline cache before the device.
        cleanupBufferPool(device);
        cleanupPipelineCache(device);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    initialized = false;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

// -----------------------------------------------------------------------------
// Pipeline Cache (Simple)
// -----------------------------------------------------------------------------

struct PipelineState {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descLayout;
    VkShaderModule shaderModule;
};

// Map shader name to pipeline state
static std::map<std::string, PipelineState> g_pipelineCache;
static std::mutex g_pipelineCacheMutex;

static void cleanupPipelineCache(VkDevice device) {
    std::lock_guard<std::mutex> lock(g_pipelineCacheMutex);
    for (auto& [name, state] : g_pipelineCache) {
        if (state.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, state.pipeline, nullptr);
        }
        if (state.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, state.layout, nullptr);
        }
        if (state.descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, state.descLayout, nullptr);
        }
        if (state.shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, state.shaderModule, nullptr);
        }
    }
    g_pipelineCache.clear();
}

static PipelineState getOrCreatePipeline(const std::string& shaderName, size_t bufferCount, size_t pushConstSize) {
    std::lock_guard<std::mutex> lock(g_pipelineCacheMutex);

    // Key on shader name + descriptor-set layout shape (buffer count) + push
    // constant size: the pipeline layout depends on all three, so caching by
    // name alone would return a layout-mismatched pipeline if a shader were ever
    // dispatched with a different binding count / push size.
    const std::string cacheKey = shaderName + "#" + std::to_string(bufferCount)
                                            + "#" + std::to_string(pushConstSize);

    if (g_pipelineCache.count(cacheKey)) {
        return g_pipelineCache[cacheKey];
    }

    VulkanContext& ctx = VulkanContext::get();
    VkDevice device = ctx.device;

    auto code = readFile(shaderName);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create shader module for " << shaderName << ": " << result << std::endl;
        throw std::runtime_error("failed to create shader module for " + shaderName);
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for(size_t i=0; i<bufferCount; ++i) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = i;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    if (result != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create descriptor set layout: " << result << std::endl;
        vkDestroyShaderModule(device, shaderModule, nullptr);
        throw std::runtime_error("failed to create descriptor set layout");
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstSize;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    if(pushConstSize > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    }

    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    if (result != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create pipeline layout: " << result << std::endl;
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        throw std::runtime_error("failed to create pipeline layout");
    }

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
    if (result != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create compute pipeline: " << result << std::endl;
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        throw std::runtime_error("failed to create compute pipeline");
    }

    PipelineState state = {computePipeline, pipelineLayout, descriptorSetLayout, shaderModule};
    g_pipelineCache[cacheKey] = state;
    return state;
}


// -----------------------------------------------------------------------------
// Helper: Run Compute Shader
// -----------------------------------------------------------------------------

// Persistent scratch-buffer pool. Every op previously did a full
// vkCreateBuffer + vkAllocateMemory on entry and vkDestroyBuffer + vkFreeMemory
// on exit; for iterative workloads (FFT stages, repeated same-size GEMM) that
// allocation churn dominates. We instead recycle HOST_VISIBLE|HOST_COHERENT
// buffers keyed by exact byte size: BufferWrapper acquires from the pool (or
// allocates on a miss) and returns the buffer to the pool on destruction.
// (True zero-copy would need VK_EXT_external_memory_host to import the Eigen
// pointer directly, which the Pi 5 v3dv driver does not expose.)
struct PooledBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
};
static std::multimap<VkDeviceSize, PooledBuffer> g_bufferPool;
static std::mutex g_bufferPoolMutex;

static bool bufferPoolAcquire(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) {
    std::lock_guard<std::mutex> lock(g_bufferPoolMutex);
    auto it = g_bufferPool.find(size);
    if (it == g_bufferPool.end()) return false;
    buffer = it->second.buffer;
    memory = it->second.memory;
    g_bufferPool.erase(it);
    return true;
}

static void bufferPoolRelease(VkDeviceSize size, VkBuffer buffer, VkDeviceMemory memory) {
    std::lock_guard<std::mutex> lock(g_bufferPoolMutex);
    g_bufferPool.emplace(size, PooledBuffer{buffer, memory});
}

static void cleanupBufferPool(VkDevice device) {
    std::lock_guard<std::mutex> lock(g_bufferPoolMutex);
    for (auto& [sz, buf] : g_bufferPool) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        vkFreeMemory(device, buf.memory, nullptr);
    }
    g_bufferPool.clear();
}

struct BufferWrapper {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;

    BufferWrapper(VkDeviceSize s) : size(s) {
        if (!bufferPoolAcquire(size, buffer, memory)) {
            createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
        }
    }
    ~BufferWrapper() {
        // Return to the pool for reuse rather than freeing (freed at cleanup).
        bufferPoolRelease(size, buffer, memory);
    }
    void mapAndCopyFrom(const void* src) {
        void* data;
        if (vkMapMemory(VulkanContext::get().device, memory, 0, size, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map buffer memory for write");
        }
        memcpy(data, src, (size_t)size);
        vkUnmapMemory(VulkanContext::get().device, memory);
    }
    void mapAndCopyTo(void* dst) {
        void* data;
        if (vkMapMemory(VulkanContext::get().device, memory, 0, size, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map buffer memory for read");
        }
        memcpy(dst, data, (size_t)size);
        vkUnmapMemory(VulkanContext::get().device, memory);
    }
};

static void run_compute(const std::string& shaderName,
                        const std::vector<BufferWrapper*>& buffers,
                        const void* pushConstData, size_t pushConstSize,
                        uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) {

    VulkanContext& ctx = VulkanContext::get();
    VkDevice device = ctx.device;

    // Validate dispatch dimensions against the device's per-dimension work-group
    // limit. v3dv (Pi 5 VideoCore VII) caps maxComputeWorkGroupCount well below
    // 2^31; exceeding it makes vkCmdDispatch silently drop the tail of the grid
    // (elements never computed). Cache the limit once — it is device-constant.
    static const VkExtent3D maxGroups = [&]() -> VkExtent3D {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
        return { props.limits.maxComputeWorkGroupCount[0],
                 props.limits.maxComputeWorkGroupCount[1],
                 props.limits.maxComputeWorkGroupCount[2] };
    }();
    if (groupCountX > maxGroups.width ||
        groupCountY > maxGroups.height ||
        groupCountZ > maxGroups.depth) {
        std::cerr << "[Vulkan] Dispatch for '" << shaderName << "' requests ("
                  << groupCountX << "," << groupCountY << "," << groupCountZ
                  << ") work groups, exceeding device limit ("
                  << maxGroups.width << "," << maxGroups.height << ","
                  << maxGroups.depth << "). Input too large for this backend.\n";
        throw std::runtime_error("compute dispatch exceeds maxComputeWorkGroupCount");
    }

    // Get Cached Pipeline
    PipelineState state = getOrCreatePipeline(shaderName, buffers.size(), pushConstSize);

    // Per-call transient Vulkan objects (descriptor pool, command buffer, fence)
    // are CACHED and reused rather than created/destroyed every dispatch — that
    // churn plus a full vkQueueWaitIdle dominated small-GPU-op latency (~4.8 ms).
    // A single mutex serializes host calls (a VkQueue must be externally
    // synchronized). The call is still synchronous — it waits on the fence
    // before returning — so the buffer-pool release in ~BufferWrapper stays safe.
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);

    static VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    static VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    static VkFence fence = VK_NULL_HANDLE;
    VkResult result;
    if (descriptorPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 16;  // >= max buffers used by any kernel
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 4;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor pool");

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = ctx.commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            throw std::runtime_error("failed to allocate command buffer");
        }
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, ctx.commandPool, 1, &commandBuffer);
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE; commandBuffer = VK_NULL_HANDLE;
            throw std::runtime_error("failed to create fence");
        }
    }

    // Reset the cached objects for this dispatch.
    vkResetDescriptorPool(device, descriptorPool, 0);
    vkResetCommandBuffer(commandBuffer, 0);
    vkResetFences(device, 1, &fence);

    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &state.descLayout;
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets");

    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());
    for(size_t i=0; i<buffers.size(); ++i) {
        bufferInfos[i].buffer = buffers[i]->buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers[i]->size;
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin command buffer");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.layout, 0, 1, &descriptorSet, 0, nullptr);
    if(pushConstSize > 0) {
        vkCmdPushConstants(commandBuffer, state.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstData);
    }
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);

    // Barrier making compute-shader writes visible to the HOST readback that
    // follows (results are read via mapped coherent memory after the fence wait).
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to end command buffer");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
        throw std::runtime_error("failed to submit queue");

    // Wait on THIS submission's fence (not the whole queue) before returning.
    result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        throw std::runtime_error("fence wait failed");
    // Cached objects are reused next call; freed by the driver at device teardown.
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------

bool is_available() {
    return VulkanContext::get().init();
}

// --- CPU/GPU offload thresholds ---
// The Pi 5 GPU (Broadcom VideoCore VII / V3D) is weaker than the 4x Cortex-A76
// CPU and shares the same LPDDR4X bus; every dispatch also pays a submit +
// map/copy + queue-wait round-trip. So GPU offload only wins for large problems.
// Below these element counts we compute on the CPU transparently (returning the
// same result), which also avoids initializing Vulkan for tiny calls.
// Measured on the Pi 5: even after caching the descriptor pool / command buffer
// / fence, a single dispatch carries ~2.9 ms of fixed submit + buffer-map/copy
// overhead. For memory-bound elementwise/dot ops the V3D never beats the 4x A76
// at any size that fits memory, so the elementwise threshold is set high (these
// effectively always run on the CPU here). The GPU only pays off for large,
// compute-heavy GEMM. Tune per board; a stronger discrete GPU wants lower values.
static constexpr size_t OPTMATH_VK_ELTWISE_MIN = 1u << 20;  // ~1M elements
static constexpr size_t OPTMATH_VK_MATMUL_MIN  = 256ull * 256 * 256;  // M*N*K

Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return Eigen::VectorXf();
    // Small problem or no GPU: compute on the CPU (correct either way).
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available()) return a + b;

    // GPU path guarded: any failure (e.g. a missing .spv, an allocation error)
    // falls back to the CPU rather than escaping as an exception past this API's
    // return-a-valid-result contract (which would std::terminate the caller).
    try {
        size_t count = a.size();
        size_t sizeBytes = count * sizeof(float);
        BufferWrapper bufA(sizeBytes);
        BufferWrapper bufB(sizeBytes);
        BufferWrapper bufOut(sizeBytes);
        bufA.mapAndCopyFrom(a.data());
        bufB.mapAndCopyFrom(b.data());
        struct { uint32_t count; } push = { (uint32_t)count };
        run_compute("vec_add.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), (uint32_t)((count + 255) / 256));
        Eigen::VectorXf res(count);
        bufOut.mapAndCopyTo(res.data());
        return res;
    } catch (const std::exception&) {
        return a + b;  // CPU fallback
    }
}

Eigen::VectorXf vulkan_vec_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    try {
    if (a.size() != b.size()) return Eigen::VectorXf();
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available()) return a - b;

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute("vec_sub.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), (uint32_t)((count + 255) / 256));

    Eigen::VectorXf res(count);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

Eigen::VectorXf vulkan_vec_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    try {
    if (a.size() != b.size()) return Eigen::VectorXf();
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available())
        return a.array() / b.array();

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute("vec_div.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), (uint32_t)((count + 255) / 256));

    Eigen::VectorXf res(count);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    try {
    if (a.size() != b.size()) return Eigen::VectorXf();
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available())
        return a.array() * b.array();

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute("vec_mul.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), (uint32_t)((count + 255) / 256));

    Eigen::VectorXf res(count);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

float vulkan_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    try {
    if (a.size() != b.size()) return 0.0f;
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available()) return a.dot(b);

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);
    uint32_t groupCount = (uint32_t)((count + 255) / 256);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(groupCount * sizeof(float)); // Partials

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute("vec_dot.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), groupCount);

    std::vector<float> partials(groupCount);
    bufOut.mapAndCopyTo(partials.data());

    float sum = 0.0f;
    for (float v : partials) sum += v;
    return sum;
    } catch (const std::exception&) { return 0.0f; }
}

float vulkan_vec_norm(const Eigen::VectorXf& a) {
    try {
    if (!is_available() || a.size() == 0) return 0.0f;
    // ||a|| = sqrt(a . a). Reuse the reduction-based dot kernel instead of
    // writing N squares to global memory and summing them on the CPU (the old
    // path was memory-bandwidth bound and slower than a single dot).
    return std::sqrt(vulkan_vec_dot(a, a));
    } catch (const std::exception&) { return 0.0f; }
}

// --- Matrix Operations ---

Eigen::MatrixXf vulkan_mat_add(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    try {
    if (a.size() == 0 || a.rows() != b.rows() || a.cols() != b.cols()) return Eigen::MatrixXf();
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available()) return a + b;

    size_t rows = a.rows();
    size_t cols = a.cols();
    size_t count = rows * cols;
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)rows, (uint32_t)cols };

    // Dispatch 2D: (rows/16, cols/16)
    uint32_t gx = (uint32_t)((rows + 15) / 16);
    uint32_t gy = (uint32_t)((cols + 15) / 16);

    run_compute("mat_add.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(rows, cols);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::MatrixXf vulkan_mat_sub(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    try {
    if (a.size() == 0 || a.rows() != b.rows() || a.cols() != b.cols()) return Eigen::MatrixXf();
    if ((size_t)a.size() < OPTMATH_VK_ELTWISE_MIN || !is_available()) return a - b;

    size_t rows = a.rows();
    size_t cols = a.cols();
    size_t count = rows * cols;
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)rows, (uint32_t)cols };

    uint32_t gx = (uint32_t)((rows + 15) / 16);
    uint32_t gy = (uint32_t)((cols + 15) / 16);

    run_compute("mat_sub.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(rows, cols);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::MatrixXf vulkan_mat_mul(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    if (a.size() == 0 || b.size() == 0 || a.cols() != b.rows()) return Eigen::MatrixXf();
    // GEMM on V3D must overcome map->copy->submit->wait->copy-back over the
    // shared bus, so it only wins for large M*N*K. Below that, the CPU (Eigen,
    // which is itself NEON-vectorized) is faster.
    {
        size_t work = (size_t)a.rows() * (size_t)b.cols() * (size_t)a.cols();
        if (work < OPTMATH_VK_MATMUL_MIN || !is_available()) return a * b;
    }

    // A: MxK, B: KxN -> C: MxN
    size_t M = a.rows();
    size_t K = a.cols();
    size_t N = b.cols();

    // GPU path guarded: fall back to the CPU (Eigen, NEON-vectorized) on any
    // Vulkan failure rather than throwing past the return-a-result contract.
    try {
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);

        BufferWrapper bufA(sizeA);
        BufferWrapper bufB(sizeB);
        BufferWrapper bufOut(sizeC);

        bufA.mapAndCopyFrom(a.data());
        bufB.mapAndCopyFrom(b.data());

        struct { uint32_t M; uint32_t K; uint32_t N; } push = { (uint32_t)M, (uint32_t)K, (uint32_t)N };

        // Select shader and tile size based on GPU
        auto& ctx = VulkanContext::get();
        if (ctx.isMaliG720) {
            uint32_t gx = (uint32_t)((M + 31) / 32);
            uint32_t gy = (uint32_t)((N + 31) / 32);
            run_compute("mat_mul_tiled_mali.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), gx, gy);
        } else {
            // Default (incl. Pi 5 VideoCore VII): 16x16 shared-memory tiled GEMM.
            uint32_t gx = (uint32_t)((M + 15) / 16);
            uint32_t gy = (uint32_t)((N + 15) / 16);
            run_compute("mat_mul_tiled.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), gx, gy);
        }

        Eigen::MatrixXf res(M, N);
        bufOut.mapAndCopyTo(res.data());
        return res;
    } catch (const std::exception&) {
        return a * b;  // CPU fallback
    }
}

Eigen::MatrixXf vulkan_mat_transpose(const Eigen::MatrixXf& a) {
    try {
    if (!is_available() || a.size() == 0) return Eigen::MatrixXf();

    size_t rows = a.rows();
    size_t cols = a.cols();
    size_t count = rows * cols;
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)rows, (uint32_t)cols };

    // Threads map to A's dimensions
    uint32_t gx = (uint32_t)((rows + 15) / 16);
    uint32_t gy = (uint32_t)((cols + 15) / 16);

    run_compute("mat_transpose.comp.spv", {&bufA, &bufOut}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(cols, rows);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::MatrixXf vulkan_mat_scale(const Eigen::MatrixXf& a, float scalar) {
    try {
    if (!is_available() || a.size() == 0) return Eigen::MatrixXf();

    size_t rows = a.rows();
    size_t cols = a.cols();
    size_t count = rows * cols;
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());

    struct { uint32_t count; float scalar; } push = { (uint32_t)count, scalar };

    // mat_scale is 1D linear
    uint32_t gx = (uint32_t)((count + 255) / 256);

    run_compute("mat_scale.comp.spv", {&bufA, &bufOut}, &push, sizeof(push), gx);

    Eigen::MatrixXf res(rows, cols);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::VectorXf vulkan_mat_vec_mul(const Eigen::MatrixXf& a, const Eigen::VectorXf& v) {
    try {
    if (!is_available() || a.size() == 0 || v.size() == 0 || a.cols() != v.size()) return Eigen::VectorXf();

    size_t rows = a.rows();
    size_t cols = a.cols();

    size_t sizeA = rows * cols * sizeof(float);
    size_t sizeV = cols * sizeof(float);
    size_t sizeOut = rows * sizeof(float);

    BufferWrapper bufA(sizeA);
    BufferWrapper bufV(sizeV);
    BufferWrapper bufOut(sizeOut);

    bufA.mapAndCopyFrom(a.data());
    bufV.mapAndCopyFrom(v.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)rows, (uint32_t)cols };
    uint32_t gx = (uint32_t)((rows + 255) / 256);

    run_compute("mat_vec_mul.comp.spv", {&bufA, &bufV, &bufOut}, &push, sizeof(push), gx);

    Eigen::VectorXf res(rows);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

Eigen::MatrixXf vulkan_mat_outer_product(const Eigen::VectorXf& u, const Eigen::VectorXf& v) {
    try {
    if (!is_available() || u.size() == 0 || v.size() == 0) return Eigen::MatrixXf();

    size_t M = u.size(); // Rows
    size_t N = v.size(); // Cols

    size_t sizeU = M * sizeof(float);
    size_t sizeV = N * sizeof(float);
    size_t sizeOut = M * N * sizeof(float);

    BufferWrapper bufU(sizeU);
    BufferWrapper bufV(sizeV);
    BufferWrapper bufOut(sizeOut);

    bufU.mapAndCopyFrom(u.data());
    bufV.mapAndCopyFrom(v.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)M, (uint32_t)N };
    uint32_t gx = (uint32_t)((M + 15) / 16);
    uint32_t gy = (uint32_t)((N + 15) / 16);

    run_compute("mat_outer_product.comp.spv", {&bufU, &bufV, &bufOut}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(M, N);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::MatrixXf vulkan_mat_elementwise_mul(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    try {
    if (!is_available() || a.size() == 0 || a.rows() != b.rows() || a.cols() != b.cols()) return Eigen::MatrixXf();

    size_t rows = a.rows();
    size_t cols = a.cols();
    size_t count = rows * cols;
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t rows; uint32_t cols; } push = { (uint32_t)rows, (uint32_t)cols };

    uint32_t gx = (uint32_t)((rows + 15) / 16);
    uint32_t gy = (uint32_t)((cols + 15) / 16);

    run_compute("mat_elementwise_mul.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(rows, cols);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::VectorXf vulkan_convolution_1d(const Eigen::VectorXf& x, const Eigen::VectorXf& k) {
    try {
    if (!is_available() || k.size() == 0 || x.size() < k.size()) return Eigen::VectorXf();

    size_t n_x = x.size();
    size_t n_k = k.size();
    size_t n_y = n_x - n_k + 1;

    BufferWrapper bufX(n_x * sizeof(float));
    BufferWrapper bufK(n_k * sizeof(float));
    BufferWrapper bufY(n_y * sizeof(float));

    bufX.mapAndCopyFrom(x.data());
    bufK.mapAndCopyFrom(k.data());

    struct { uint32_t n_x; uint32_t n_h; } push = { (uint32_t)n_x, (uint32_t)n_k };
    // Using new file name
    run_compute("convolution_1d.comp.spv", {&bufX, &bufK, &bufY}, &push, sizeof(push), (uint32_t)((n_y + 255) / 256));

    Eigen::VectorXf res(n_y);
    bufY.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

Eigen::MatrixXf vulkan_convolution_2d(const Eigen::MatrixXf& x, const Eigen::MatrixXf& k) {
    try {
    if (!is_available() || k.size() == 0 || x.rows() < k.rows() || x.cols() < k.cols()) return Eigen::MatrixXf();

    size_t H_in = x.rows();
    size_t W_in = x.cols();
    size_t K_h = k.rows();
    size_t K_w = k.cols();

    size_t H_out = H_in - K_h + 1;
    size_t W_out = W_in - K_w + 1;

    size_t sizeX = H_in * W_in * sizeof(float);
    size_t sizeK = K_h * K_w * sizeof(float);
    size_t sizeY = H_out * W_out * sizeof(float);

    BufferWrapper bufX(sizeX);
    BufferWrapper bufK(sizeK);
    BufferWrapper bufY(sizeY);

    bufX.mapAndCopyFrom(x.data());
    bufK.mapAndCopyFrom(k.data());

    struct { uint32_t H_in; uint32_t W_in; uint32_t K_h; uint32_t K_w; } push = {
        (uint32_t)H_in, (uint32_t)W_in, (uint32_t)K_h, (uint32_t)K_w
    };

    uint32_t gx = (uint32_t)((H_out + 15) / 16);
    uint32_t gy = (uint32_t)((W_out + 15) / 16);

    run_compute("convolution_2d.comp.spv", {&bufX, &bufK, &bufY}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(H_out, W_out);
    bufY.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

Eigen::VectorXf vulkan_correlation_1d(const Eigen::VectorXf& x, const Eigen::VectorXf& k) {
    try {
    if (!is_available() || k.size() == 0 || x.size() < k.size()) return Eigen::VectorXf();

    size_t n_x = x.size();
    size_t n_k = k.size();
    size_t n_y = n_x - n_k + 1;

    BufferWrapper bufX(n_x * sizeof(float));
    BufferWrapper bufK(n_k * sizeof(float));
    BufferWrapper bufY(n_y * sizeof(float));

    bufX.mapAndCopyFrom(x.data());
    bufK.mapAndCopyFrom(k.data());

    struct { uint32_t n_x; uint32_t n_h; } push = { (uint32_t)n_x, (uint32_t)n_k };
    run_compute("correlation_1d.comp.spv", {&bufX, &bufK, &bufY}, &push, sizeof(push), (uint32_t)((n_y + 255) / 256));

    Eigen::VectorXf res(n_y);
    bufY.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

Eigen::MatrixXf vulkan_correlation_2d(const Eigen::MatrixXf& x, const Eigen::MatrixXf& k) {
    try {
    if (!is_available() || k.size() == 0 || x.rows() < k.rows() || x.cols() < k.cols()) return Eigen::MatrixXf();

    size_t H_in = x.rows();
    size_t W_in = x.cols();
    size_t K_h = k.rows();
    size_t K_w = k.cols();

    size_t H_out = H_in - K_h + 1;
    size_t W_out = W_in - K_w + 1;

    size_t sizeX = H_in * W_in * sizeof(float);
    size_t sizeK = K_h * K_w * sizeof(float);
    size_t sizeY = H_out * W_out * sizeof(float);

    BufferWrapper bufX(sizeX);
    BufferWrapper bufK(sizeK);
    BufferWrapper bufY(sizeY);

    bufX.mapAndCopyFrom(x.data());
    bufK.mapAndCopyFrom(k.data());

    struct { uint32_t H_in; uint32_t W_in; uint32_t K_h; uint32_t K_w; } push = {
        (uint32_t)H_in, (uint32_t)W_in, (uint32_t)K_h, (uint32_t)K_w
    };

    uint32_t gx = (uint32_t)((H_out + 15) / 16);
    uint32_t gy = (uint32_t)((W_out + 15) / 16);

    run_compute("correlation_2d.comp.spv", {&bufX, &bufK, &bufY}, &push, sizeof(push), gx, gy);

    Eigen::MatrixXf res(H_out, W_out);
    bufY.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::MatrixXf(); }
}

// Helper for reduction
static float run_reduction(const Eigen::VectorXf& a, const std::string& shaderName, float initialVal, float (*cpuReduce)(float, float), uint32_t workgroupSize = 256) {
    if (!is_available() || a.size() == 0) return initialVal;

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);
    uint32_t groupCount = (uint32_t)((count + workgroupSize - 1) / workgroupSize);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufOut(groupCount * sizeof(float));

    bufA.mapAndCopyFrom(a.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute(shaderName, {&bufA, &bufOut}, &push, sizeof(push), groupCount);

    std::vector<float> partials(groupCount);
    bufOut.mapAndCopyTo(partials.data());

    // Final reduction on CPU for simplicity
    float result = partials[0];
    for (size_t i = 1; i < partials.size(); ++i) {
        result = cpuReduce(result, partials[i]);
    }
    return result;
}

static ReduceBackend g_reduceBackend = ReduceBackend::Auto;
void set_reduce_backend(ReduceBackend b) { g_reduceBackend = b; }
ReduceBackend get_reduce_backend() { return g_reduceBackend; }
bool subgroup_reduce_available() { return VulkanContext::get().subgroupCanReduce; }

float vulkan_reduce_sum(const Eigen::VectorXf& a) {
    try {
    auto& ctx = VulkanContext::get();
    if (ctx.isMaliG720) {
        return run_reduction(a, "reduce_sum_mali.comp.spv", 0.0f, [](float x, float y){ return x + y; }, 1024);
    }
    // Prefer the subgroup-shuffle kernel when the device supports it (V3D does);
    // the barrier tree remains the portable fallback. Both use a 256-wide group.
    const bool useSubgroup =
        (g_reduceBackend == ReduceBackend::Subgroup) ||
        (g_reduceBackend == ReduceBackend::Auto && ctx.subgroupCanReduce);
    const char* shader = useSubgroup ? "reduce_sum_subgroup.comp.spv" : "reduce_sum.comp.spv";
    return run_reduction(a, shader, 0.0f, [](float x, float y){ return x + y; });
    } catch (const std::exception&) { return 0.0f; }
}

float vulkan_reduce_max(const Eigen::VectorXf& a) {
    try {
    if (a.size() == 0) return 0.0f;
    return run_reduction(a, "reduce_max.comp.spv", a[0], [](float x, float y){ return std::max(x, y); });
    } catch (const std::exception&) { return 0.0f; }
}

float vulkan_reduce_min(const Eigen::VectorXf& a) {
    try {
    if (a.size() == 0) return 0.0f;
    return run_reduction(a, "reduce_min.comp.spv", a[0], [](float x, float y){ return std::min(x, y); });
    } catch (const std::exception&) { return 0.0f; }
}

Eigen::VectorXf vulkan_scan_prefix_sum(const Eigen::VectorXf& a) {
    try {
    if (!is_available() || a.size() == 0) return Eigen::VectorXf();

    // Note: The shader `scan_prefix_sum.comp.glsl` is a single-block scan.
    // It only works correctly if a.size() <= 256.
    // For larger sizes, we'd need a multi-pass approach (reduce-scan-downsweep).
    // For now, we will run it as-is, but warn or fallback if N > 256.

    if (a.size() > 256) {
        std::cerr << "[Vulkan] Warning: scan_prefix_sum only supports size <= 256 in this version. Falling back to CPU.\n";
        Eigen::VectorXf res(a.size());
        float sum = 0.0f;
        for (int i=0; i<a.size(); ++i) {
            float v = a[i];
            res[i] = sum; // Exclusive
            sum += v;
        }
        return res;
    }

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());

    struct { uint32_t count; } push = { (uint32_t)count };
    run_compute("scan_prefix_sum.comp.spv", {&bufA, &bufOut}, &push, sizeof(push), 1);

    Eigen::VectorXf res(count);
    bufOut.mapAndCopyTo(res.data());
    return res;
    } catch (const std::exception&) { return Eigen::VectorXf(); }
}

// Bit reversal helper
static void bit_reverse_copy(const Eigen::VectorXf& in, Eigen::VectorXf& out) {
    size_t N = in.size() / 2; // Complex pairs
    uint32_t levels = 0;
    for (size_t tmp = N; tmp > 1; tmp >>= 1) levels++;
    for (size_t i = 0; i < N; ++i) {
        // Reverse bits of i
        size_t r = 0;
        size_t temp = i;
        for (uint32_t j = 0; j < levels; ++j) {
            r = (r << 1) | (temp & 1);
            temp >>= 1;
        }
        out[2*r] = in[2*i];
        out[2*r+1] = in[2*i+1];
    }
}

void vulkan_fft_radix2(Eigen::VectorXf& data, bool inverse) {
    try {
    if (!is_available() || data.size() == 0 || (data.size() % 2 != 0)) return;

    // data is interleaved complex. N points -> 2*N floats.
    size_t N = data.size() / 2;
    // N must be power of 2
    if ((N & (N - 1)) != 0) {
        std::cerr << "[Vulkan] FFT Radix-2 requires size power of 2\n";
        return;
    }

    // Cooley-Tukey iterative usually needs bit-reversal first.
    // We do bit-reversal on CPU for simplicity (or can add a shader).
    Eigen::VectorXf reversed(data.size());
    bit_reverse_copy(data, reversed);

    // Copy to GPU
    size_t sizeBytes = data.size() * sizeof(float);
    BufferWrapper buf(sizeBytes);
    buf.mapAndCopyFrom(reversed.data());

    uint32_t stages = 0;
    for (size_t tmp = N; tmp > 1; tmp >>= 1) stages++;
    std::string shader = inverse ? "ifft_radix2.comp.spv" : "fft_radix2.comp.spv";

    for (uint32_t s = 0; s < stages; ++s) {
        // We launch N/2 threads (one per butterfly)
        struct { uint32_t n; uint32_t stage; uint32_t invert; } push = {
            (uint32_t)N, s, (uint32_t)(inverse ? 1 : 0)
        };
        uint32_t groupCount = (uint32_t)((N/2 + 255) / 256);
        run_compute(shader, {&buf}, &push, sizeof(push), groupCount);

        // Ensure barrier between stages?
        // run_compute submits and waits for idle, so yes, it's synchronized.
    }

    // Read back
    buf.mapAndCopyTo(data.data());

    // Note: follows FFTW/cuFFT convention — caller is responsible for 1/N normalization on IFFT
    } catch (const std::exception&) { return; }
}

void vulkan_fft_radix4(Eigen::VectorXf& data, bool inverse) {
    try {
    // The radix-4 GPU path requires base-4 digit reversal to permute its input,
    // but only base-2 bit reversal is implemented, which produces numerically
    // incorrect results. Until the base-4 permutation (and shader) are validated
    // against a golden FFT, delegate to the verified radix-2 Cooley-Tukey path,
    // which is correct for any power-of-2 size (including all powers of 4).
    vulkan_fft_radix2(data, inverse);
    } catch (const std::exception&) { return; }
}

#else

// Stubs when Vulkan is disabled
bool is_available() { return false; }
VulkanContext& VulkanContext::get() { static VulkanContext ctx; return ctx; }
bool VulkanContext::init() { return false; }
void VulkanContext::cleanup() {}

Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::VectorXf vulkan_vec_sub(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::VectorXf vulkan_vec_div(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
float           vulkan_vec_dot(const Eigen::VectorXf&, const Eigen::VectorXf&) { return 0.0f; }
float           vulkan_vec_norm(const Eigen::VectorXf&) { return 0.0f; }

Eigen::MatrixXf vulkan_mat_add(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_sub(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_mul(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_transpose(const Eigen::MatrixXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_scale(const Eigen::MatrixXf&, float) { return {}; }

Eigen::VectorXf vulkan_mat_vec_mul(const Eigen::MatrixXf&, const Eigen::VectorXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_outer_product(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::MatrixXf vulkan_mat_elementwise_mul(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }

Eigen::VectorXf vulkan_convolution_1d(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::MatrixXf vulkan_convolution_2d(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }
Eigen::VectorXf vulkan_correlation_1d(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::MatrixXf vulkan_correlation_2d(const Eigen::MatrixXf&, const Eigen::MatrixXf&) { return {}; }

void set_reduce_backend(ReduceBackend) {}
ReduceBackend get_reduce_backend() { return ReduceBackend::Auto; }
bool subgroup_reduce_available() { return false; }
float vulkan_reduce_sum(const Eigen::VectorXf&) { return 0.0f; }
float vulkan_reduce_max(const Eigen::VectorXf&) { return 0.0f; }
float vulkan_reduce_min(const Eigen::VectorXf&) { return 0.0f; }
Eigen::VectorXf vulkan_scan_prefix_sum(const Eigen::VectorXf&) { return {}; }

void vulkan_fft_radix2(Eigen::VectorXf&, bool) {}
void vulkan_fft_radix4(Eigen::VectorXf&, bool) {}

#endif // OPTMATH_USE_VULKAN

}
}
