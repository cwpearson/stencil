# add -Wnull-deference for g++ >= 6
# add -Wrestrict g++ >= 7
# add -Wconversion
# add -Wduplicated-cond for g++ >= 6
# add -Wduplicated-branches for g++ >= 7

macro(add_args tgt)
target_compile_options(
  ${tgt}
  PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall;
    -Wextra;
    -Wpedantic;
    -Wcast-align;
    -Wdisabled-optimization;
    -Winit-self;
    -Wlogical-op;
    -Wmissing-include-dirs;
    -Woverloaded-virtual;
    -Wpointer-arith;
    -Wshadow;
    -Wstrict-aliasing;
    -Wswitch-enum;
    -Wundef;
    -Wvla;
    -Wformat=2;
    -mtune=native;
  >
)


## Add CUDA flags
target_compile_options(
  ${tgt}
  PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:
    --Wno-deprecated-gpu-targets;
    --expt-extended-lambda;
    -Xcompiler=-Wall;
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wdisabled-optimization;
    -Xcompiler=-Wformat=2;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wlogical-op;
    -Xcompiler=-Wmissing-include-dirs;
    -Xcompiler=-Woverloaded-virtual;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Wshadow;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wvla;
    -Xptxas=-v;
    -Xcompiler=-mtune=native;
    --use_fast_math;
  >
)


if (CXX_HAS_MARCH)
    target_compile_options(
      ${tgt}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
  elseif (CXX_HAS_MCPU)
    target_compile_options(
      ${tgt}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcpu=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>
    )
  endif()

target_include_directories(${tgt} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
# we require separable compilation for stencil library
set_property(TARGET ${tgt} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endmacro()


add_executable(astaroth-sim astaroth_sim.cu)
target_link_libraries(astaroth-sim stencil::stencil)
add_args(astaroth-sim)

add_executable(bench-mpi bench_mpi.cu statistics.cpp)
set_target_properties(bench-mpi PROPERTIES ENABLE_EXPORTS ON) # better back trace
target_link_libraries(bench-mpi stencil::stencil)
add_args(bench-mpi)

add_executable(jacobi3d jacobi3d.cu statistics.cpp)
target_link_libraries(jacobi3d stencil::stencil)
add_args(jacobi3d)

add_executable(jacobi3d-strong jacobi3d_strong.cu statistics.cpp)
target_link_libraries(jacobi3d-strong stencil::stencil)
add_args(jacobi3d-strong)

add_executable(exchange-weak exchange_weak.cu statistics.cpp)
target_link_libraries(exchange-weak stencil::stencil)
add_args(exchange-weak)

add_executable(exchange-strong exchange_strong.cu statistics.cpp)
target_link_libraries(exchange-strong stencil::stencil)
add_args(exchange-strong)

add_executable(pingpong pingpong.cu)
target_link_libraries(pingpong stencil::stencil)
add_args(pingpong)

add_executable(measure-buf-exchange measure_buf_exchange.cu)
target_link_libraries(measure-buf-exchange stencil::stencil)
add_args(measure-buf-exchange)

add_executable(bench-exchange bench_exchange.cu statistics.cpp)
target_link_libraries(bench-exchange stencil::stencil)
add_args(bench-exchange)

add_executable(bench-qap bench_qap.cu)
target_link_libraries(bench-qap stencil::stencil)
add_args(bench-qap)

add_executable(bench-alltoallv bench_alltoallv.cu)
target_link_libraries(bench-alltoallv stencil::stencil)
add_args(bench-alltoallv)

add_executable(bench-pack bench_pack.cu)
target_link_libraries(bench-pack stencil::stencil)
add_args(bench-pack)

add_executable(machine-info machine_info.cu)
target_link_libraries(machine-info stencil::stencil)
add_args(machine-info)