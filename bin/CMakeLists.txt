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
    --use_fast_math;
  >
)

target_include_directories(${tgt} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
endmacro()


add_executable(astaroth-sim astaroth_sim.cu)
target_link_libraries(astaroth-sim stencil::stencil)
add_args(astaroth-sim)

add_executable(jacobi3d jacobi3d.cu statistics.cpp)
target_link_libraries(jacobi3d stencil::stencil)
add_args(jacobi3d)

add_executable(jacobi3d-strong jacobi3d_strong.cu statistics.cpp)
target_link_libraries(jacobi3d-strong stencil::stencil)
add_args(jacobi3d-strong)

add_executable(weak weak.cu)
target_link_libraries(weak stencil::stencil)
add_args(weak)

add_executable(weak-exchange weak_exchange.cu)
target_link_libraries(weak-exchange stencil::stencil)
add_args(weak-exchange)

add_executable(strong strong.cu)
target_link_libraries(strong stencil::stencil)
add_args(strong)

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
