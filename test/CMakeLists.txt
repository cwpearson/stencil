if (CMAKE_CUDA_COMPILER)
    add_executable(test_cuda test_cuda_main.cu test_local_domain.cu test_cuda_pack.cu test_any_tx.cu)
    target_include_directories(test_cuda SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
    target_link_libraries(test_cuda stencil)
    add_test(NAME test_cuda COMMAND test_cuda -a)
endif()


add_executable(test_cpu test_cpu_main.cpp)
target_include_directories(test_cpu SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(test_cpu stencil)
add_test(NAME test_cpu COMMAND test_cpu -a)