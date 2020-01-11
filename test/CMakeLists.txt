if (CMAKE_CUDA_COMPILER)
    add_executable(test_cuda test_cuda_main.cu 
    test_any_tx.cu
      test_cuda_array.cu 
      test_cuda_gpu_topo.cu 
      test_cuda_local_domain.cu 
      test_cuda_pack.cu 
      test_cuda_rcstream.cu 
    )
    target_include_directories(test_cuda SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
    target_link_libraries(test_cuda stencil)
    add_test(NAME test_cuda COMMAND cuda-memcheck test_cuda -a)
endif()

if (CMAKE_CUDA_COMPILER AND MPI_FOUND)
    add_executable(test_cuda_mpi test_cuda_mpi_main.cu )
    target_include_directories(test_cuda_mpi SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
    target_link_libraries(test_cuda_mpi stencil)
    add_test(NAME test_cuda_mpi COMMAND mpirun -n 2 test_cuda_mpi -a)
endif()


add_executable(test_cpu test_cpu_main.cpp
  test_cpu_array.cpp
  test_cpu_radius.cpp
  test_cpu_tx.cpp
)
target_include_directories(test_cpu SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(test_cpu stencil)
add_test(NAME test_cpu COMMAND test_cpu -a)