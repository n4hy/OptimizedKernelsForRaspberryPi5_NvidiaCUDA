add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE OptMathKernels::OptMathKernels)
add_test(NAME test_basic COMMAND test_basic)
