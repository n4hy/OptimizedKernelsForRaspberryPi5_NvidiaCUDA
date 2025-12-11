#include "optmath/vulkan_backend.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <filesystem>
#include <map>

namespace optmath {
namespace vulkan {

// Only compile implementation if Vulkan is enabled
#ifdef OPTMATH_USE_VULKAN

// Utility: Read file
static std::vector<char> readFile(const std::string& filename) {
    // Try multiple paths:
    // 1. Current directory
    // 2. Relative to executable (simplistic check for build dir structure)
    // 3. System install path (e.g. /usr/local/share/optmathkernels/shaders/)

    std::vector<std::string> paths = {
        filename,
        "../src/" + filename, // For build/examples/demo -> build/src/
        "/usr/local/share/optmathkernels/shaders/" + filename,
        "/usr/share/optmathkernels/shaders/" + filename
    };

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);
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
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

// -----------------------------------------------------------------------------
// VulkanContext
// -----------------------------------------------------------------------------

VulkanContext& VulkanContext::get() {
    static VulkanContext instance;
    return instance;
}

bool VulkanContext::init() {
    if (initialized) return true;

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

    // Pick the first one for now
    physicalDevice = devices[0];

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

    initialized = true;
    return true;
}

void VulkanContext::cleanup() {
    if (device) {
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

static PipelineState getOrCreatePipeline(const std::string& shaderName, size_t bufferCount, size_t pushConstSize) {
    if (g_pipelineCache.count(shaderName)) {
        return g_pipelineCache[shaderName];
    }

    VulkanContext& ctx = VulkanContext::get();
    VkDevice device = ctx.device;

    auto code = readFile(shaderName);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
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
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

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
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

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
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);

    PipelineState state = {computePipeline, pipelineLayout, descriptorSetLayout, shaderModule};
    g_pipelineCache[shaderName] = state;
    return state;
}


// -----------------------------------------------------------------------------
// Helper: Run Compute Shader
// -----------------------------------------------------------------------------

struct BufferWrapper {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;

    BufferWrapper(VkDeviceSize s) : size(s) {
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    }
    ~BufferWrapper() {
        vkDestroyBuffer(VulkanContext::get().device, buffer, nullptr);
        vkFreeMemory(VulkanContext::get().device, memory, nullptr);
    }
    void mapAndCopyFrom(const void* src) {
        void* data;
        vkMapMemory(VulkanContext::get().device, memory, 0, size, 0, &data);
        memcpy(data, src, (size_t)size);
        vkUnmapMemory(VulkanContext::get().device, memory);
    }
    void mapAndCopyTo(void* dst) {
        void* data;
        vkMapMemory(VulkanContext::get().device, memory, 0, size, 0, &data);
        memcpy(dst, data, (size_t)size);
        vkUnmapMemory(VulkanContext::get().device, memory);
    }
};

static void run_compute(const std::string& shaderName,
                        const std::vector<BufferWrapper*>& buffers,
                        const void* pushConstData, size_t pushConstSize,
                        uint32_t groupCountX) {

    VulkanContext& ctx = VulkanContext::get();
    VkDevice device = ctx.device;

    // Get Cached Pipeline
    PipelineState state = getOrCreatePipeline(shaderName, buffers.size(), pushConstSize);

    // Descriptors (Created per call for simplicity in handling buffer ptrs,
    // but could also be cached if buffers reused. For this level of optimization, pool management is acceptable overhead vs pipeline creation).

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = (uint32_t)buffers.size();

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &state.descLayout;

    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

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

    // Command Buffer
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.layout, 0, 1, &descriptorSet, 0, nullptr);
    if(pushConstSize > 0) {
        vkCmdPushConstants(commandBuffer, state.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstData);
    }
    vkCmdDispatch(commandBuffer, groupCountX, 1, 1);
    vkEndCommandBuffer(commandBuffer);

    // Submit and Wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.computeQueue);

    // Cleanup Command Buffer & Descriptor Pool (Pipeline is cached)
    vkFreeCommandBuffers(device, ctx.commandPool, 1, &commandBuffer);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    // Note: We do not destroy pipeline/layout here as they are cached.
    // They will leak at shutdown in this simple implementation, which is acceptable for a singleton.
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------

bool is_available() {
    return VulkanContext::get().init();
}

Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (!is_available() || a.size() != b.size()) return Eigen::VectorXf();

    size_t count = a.size();
    size_t sizeBytes = count * sizeof(float);

    BufferWrapper bufA(sizeBytes);
    BufferWrapper bufB(sizeBytes);
    BufferWrapper bufOut(sizeBytes);

    bufA.mapAndCopyFrom(a.data());
    bufB.mapAndCopyFrom(b.data());

    struct { uint32_t count; } push = { (uint32_t)count };

    // We assume the spv file is in current dir for simplicity or a fixed path
    run_compute("vec_add.comp.spv", {&bufA, &bufB, &bufOut}, &push, sizeof(push), (uint32_t)((count + 255) / 256));

    Eigen::VectorXf res(count);
    bufOut.mapAndCopyTo(res.data());
    return res;
}

Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (!is_available() || a.size() != b.size()) return Eigen::VectorXf();

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
}

float vulkan_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (!is_available() || a.size() != b.size()) return 0.0f;

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
}

Eigen::VectorXf vulkan_conv1d(const Eigen::VectorXf& x, const Eigen::VectorXf& h) {
    if (!is_available() || x.size() < h.size()) return Eigen::VectorXf();

    size_t n_x = x.size();
    size_t n_h = h.size();
    size_t n_y = n_x - n_h + 1;

    BufferWrapper bufX(n_x * sizeof(float));
    BufferWrapper bufH(n_h * sizeof(float));
    BufferWrapper bufY(n_y * sizeof(float));

    bufX.mapAndCopyFrom(x.data());
    bufH.mapAndCopyFrom(h.data());

    struct { uint32_t n_x; uint32_t n_h; } push = { (uint32_t)n_x, (uint32_t)n_h };
    run_compute("conv1d.comp.spv", {&bufX, &bufH, &bufY}, &push, sizeof(push), (uint32_t)((n_y + 255) / 256));

    Eigen::VectorXf res(n_y);
    bufY.mapAndCopyTo(res.data());
    return res;
}

#else

// Stubs when Vulkan is disabled
bool is_available() { return false; }
VulkanContext& VulkanContext::get() { static VulkanContext ctx; return ctx; }
bool VulkanContext::init() { return false; }
void VulkanContext::cleanup() {}

Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }
float           vulkan_vec_dot(const Eigen::VectorXf&, const Eigen::VectorXf&) { return 0.0f; }
Eigen::VectorXf vulkan_conv1d(const Eigen::VectorXf&, const Eigen::VectorXf&) { return {}; }

#endif // OPTMATH_USE_VULKAN

}
}
