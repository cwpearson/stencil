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


add_executable(astaroth astaroth.cu kernels.cu)
target_link_libraries(astaroth stencil::stencil)
add_args(astaroth)