# add -Wnull-deference for g++ >= 6
# add -Wrestrict g++ >= 7
# add -Wconversion

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
    -Wduplicated-branches;
    -Wduplicated-cond;
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
    -Xcompiler=-Wduplicated-branches;
    -Xcompiler=-Wduplicated-cond;
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
  >
)
endmacro()


add_executable(astaroth-sim astaroth_sim.cu)
target_include_directories(astaroth-sim SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(astaroth-sim stencil::stencil)
add_args(astaroth-sim)

add_executable(weak weak.cu)
target_link_libraries(weak stencil::stencil)
add_args(weak)

add_executable(strong strong.cu)
target_link_libraries(strong stencil::stencil)
add_args(strong)

add_executable(pingpong pingpong.cu)
target_link_libraries(pingpong stencil::stencil)
add_args(pingpong)

add_executable(bench-qap bench_qap.cu)
target_link_libraries(bench-qap stencil::stencil)
add_args(bench-qap)