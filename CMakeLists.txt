# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories
# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(stencil LANGUAGES CXX VERSION 0.1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
include(CheckCXXCompilerFlag)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

option(USE_MPI "compile with MPI support" ON)
option(USE_CUDA "compile with CUDA support" ON)
option(USE_CUDA_AWARE_MPI "assume CUDA-aware MPI support" OFF)
option(SETUP_STATS "track statistics that may increase setup time" ON)
option(EXCHANGE_STATS "track statistics that may increase exchange time" OFF)
option(USE_CUDA_GRAPH "use CUDA graph API to accelerate calls" ON)

# Set a log level if none was specified
if(NOT DEFINED STENCIL_OUTPUT_LEVEL)
message(STATUS "STENCIL_OUTPUT_LEVEL not defined, setting to INFO")
set(STENCIL_OUTPUT_LEVEL INFO CACHE STRING "one of SPEW DEBUG INFO WARN ERROR FATAL")
endif()

# Allow nvcc to choose CUDA architectures if none defined
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake")

get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

if (USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(STATUS "No CUDA support")
  endif()
endif()

if(USE_MPI)
  find_package(MPI)
endif()

add_subdirectory(src)

# only static libraries are suported by the device linker?
# so I think we need STATIC with CUDA_SEPARABLE_COMPILATION
add_library(stencil STATIC ${STENCIL_SOURCES})
add_library(stencil::stencil ALIAS stencil)
set_property(TARGET stencil PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if (STENCIL_OUTPUT_LEVEL MATCHES SPEW)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=5)
elseif (STENCIL_OUTPUT_LEVEL MATCHES DEBUG)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=4)
elseif (STENCIL_OUTPUT_LEVEL MATCHES INFO)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=3)
elseif (STENCIL_OUTPUT_LEVEL MATCHES WARN)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=2)
elseif (STENCIL_OUTPUT_LEVEL MATCHES ERROR)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=1)
elseif (STENCIL_OUTPUT_LEVEL MATCHES FATAL)
  target_compile_definitions(stencil PUBLIC STENCIL_OUTPUT_LEVEL=0)
else()
  message(FATAL_ERROR "unrecognized value of STENCIL_OUTPUT_LEVEL: "  ${STENCIL_OUTPUT_LEVEL}  ", must be one of SPEW DEBUG INFO WARN ERROR FATAL")
endif()

set_property(TARGET stencil PROPERTY CXX_STANDARD 11)
set_property(TARGET stencil PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET stencil PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET stencil PROPERTY CUDA_STANDARD 11)
set_property(TARGET stencil PROPERTY CUDA_STANDARD_REQUIRED ON)

target_compile_options(stencil PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
 -Wall
 -Wextra
 -Wcast-align;
 -Wdisabled-optimization;
 -Wformat=2;
 -Winit-self;
 -Wlogical-op;
 -Wmissing-include-dirs;
 -Woverloaded-virtual;
 -Wpointer-arith;
 -Wshadow;
 -Wstrict-aliasing;
 -Wswitch-enum;
 -Wvla;
 >
)

target_compile_options(stencil PRIVATE
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
>
)

check_cxx_compiler_flag(-march=native CXX_HAS_MARCH)
check_cxx_compiler_flag(-mcpu=native CXX_HAS_MCPU)
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    stencil
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
     -G;
     -Xcompiler=-fno-omit-frame-pointer;
    >
    $<$<COMPILE_LANGUAGE:CXX>:
     -fno-omit-frame-pointer;
     -g;
    >
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    stencil
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
  # always apply mtune
  target_compile_options(
    stencil
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mtune=native>
    $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
  )
  # prefer march=native, but fallback to mcpu=native if needed
  if (CXX_HAS_MARCH)
    message(STATUS "-march=native detected")
    target_compile_options(
      stencil
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
  elseif (CXX_HAS_MCPU)
    message(STATUS "-mcpu=native instead of -march=native")
    target_compile_options(
      stencil
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcpu=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>
    )
  endif()
endif()

if (USE_CUDA_AWARE_MPI)
  message(STATUS "Using CUDA-Aware MPI")
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_CUDA_AWARE_MPI=1)
else()
	message(STATUS "NOT using CUDA-Aware MPI (USE_CUDA_AWARE_MPI=ON to enable)")
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_CUDA_AWARE_MPI=0)
endif()

if (EXCHANGE_STATS)
  message(STATUS "EXCHANGE_STATS=ON, compiling with -DSTENCIL_EXCHANGE_STATS. Exchange may be slowed.")
  target_compile_definitions(stencil PUBLIC -DSTENCIL_EXCHANGE_STATS=1)
else()
  message(STATUS "EXCHANGE_STATS=OFF")
endif()

if (SETUP_STATS)
  message(STATUS "SETUP_STATS=ON, compiling with -DSTENCIL_SETUP_STATS. Setup may be slowed.")
  target_compile_definitions(stencil PUBLIC -DSTENCIL_SETUP_STATS=1)
else()
  message(STATUS "SETUP_STATS=OFF")
endif()

if (USE_CUDA_GRAPH)
  message(STATUS "USE_CUDA_GRAPH=ON, compiling with -DSTENCIL_USE_CUDA_GRAPH")
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_CUDA_GRAPH)
endif()

if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_CUDA=1)
  message(STATUS "CUDA found, compiling with STENCIL_USE_CUDA=1")

  set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
  find_library(NVTOOLSEXT_LIB nvToolsExt
              HINTS
              "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
              "${CUDA_TOOLKIT_ROOT_DIR}/lib"
              "${CUDA_TOOLKIT_ROOT_DIR}"
              )
  message(STATUS "nvToolsExt found: ${NVTOOLSEXT_LIB}")

else()
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_CUDA=0)
  message(STATUS "CUDA not found, compiling with STENCIL_USE_CUDA=0")
endif()

## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
## causes device linking with -pthread, even as of 3.15.0-rc1
## https://gitlab.kitware.com/cmake/cmake/issues/18897
## Add MPI flags
if(MPI_FOUND)
  target_include_directories(stencil SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})

  target_link_libraries(stencil PUBLIC ${MPI_CXX_LIBRARIES})
  # target_link_libraries(stencil INTERFACE MPI::MPI_CXX)
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_MPI=1)
  message(STATUS "MPI found, compiling with STENCIL_USE_MPI=1")
  message(STATUS "MPI_CXX_LIBRARIES=" ${MPI_CXX_LIBRARIES})
  message(STATUS "MPIEXEC_EXECUTABLE=" ${MPIEXEC_EXECUTABLE})
  add_custom_target(mpiexec_link ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${MPIEXEC_EXECUTABLE} mpiexec)
else()
  target_compile_definitions(stencil PUBLIC -DSTENCIL_USE_MPI=0)
  message(STATUS "MPI not found,  compiling with STENCIL_USE_MPI=0")
endif()

## Add include directories
target_include_directories(stencil SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(stencil PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(stencil PUBLIC nvidia-ml)
target_link_libraries(stencil PUBLIC ${NVTOOLSEXT_LIB})

# "this command should be in the source directory root for CTest to find the test file"
enable_testing() 

add_subdirectory(test)
add_subdirectory(astaroth)
add_subdirectory(bin)

