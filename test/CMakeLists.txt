add_executable(test_cpu test_cpu_main.cpp
  test_cpu_accessor.cpp
  test_cpu_array.cpp
  test_cpu_mat2d.cpp
  test_cpu_numeric.cpp
  test_cpu_partition.cpp
  test_cpu_qap.cpp
  test_cpu_radius.cpp
  test_cpu_tx.cpp
)
set_source_files_properties(test_cpu_partition.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(test_cpu SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(test_cpu stencil)
add_test(NAME test_cpu COMMAND ${MPIEXEC_EXECUTABLE} -n 1 test_cpu -a)

if (CMAKE_CUDA_COMPILER)
    add_executable(test_cuda test_cuda_main.cu 
      test_cuda_align.cu
      test_cuda_allocator.cu
      test_cuda_array.cu 
      test_derivative.cu
      test_exchange.cu
      test_cuda_gpu_topo.cu 
      test_cuda_local_domain.cu 
      test_cuda_pack.cu 
      test_cuda_packer.cu
      test_cuda_rcstream.cu
      test_cuda_translate.cu
      test_cuda_translate_kernel.cu
    )
    target_include_directories(test_cuda SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
    target_link_libraries(test_cuda stencil)
    set_property(TARGET test_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME test_cuda COMMAND ${MPIEXEC_EXECUTABLE} -n 1 cuda-memcheck --report-api-errors no test_cuda -a)
endif()

if (CMAKE_CUDA_COMPILER AND MPI_FOUND)
    add_executable(test_cuda_mpi test_cuda_mpi_main.cu
      test_cuda_mpi_distributed_domain.cu
      test_cuda_mpi_exchange.cu
      test_cuda_mpi_colocatedtx.cu
      test_cuda_mpi_tx_ipc.cpp
      test_cuda_mpi_cudaipc.cu
      test_cuda_aware_mpi.cu
    )
    target_include_directories(test_cuda_mpi SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
    target_link_libraries(test_cuda_mpi stencil)
    set_property(TARGET test_cuda_mpi PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME test_cuda_mpi COMMAND ${MPIEXEC_EXECUTABLE} -n 2 cuda-memcheck --report-api-errors no test_cuda_mpi -a)
endif()



