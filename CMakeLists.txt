# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories
# 3.13 doesn't detect openmpi in 18.04 on travis
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(stencil LANGUAGES CXX VERSION 0.1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

option(USE_NUMA "compile with NUMA support" ON)
option(USE_MPI "compile with MPI support" ON)
option(USE_CUDA "compile with CUDA support" ON)
option(USE_CUDA_AWARE_MPI "assume CUDA-aware MPI support" OFF)
option(MEASURE_TIMINGS "measure phase elapsed time in library" OFF)
option(LOUD "print a lot of things" ON)

if(NOT DEFINED STENCIL_OUTPUT_LEVEL)
message(STATUS "STENCIL_OUTPUT_LEVEL not defined, setting to INFO")
set(STENCIL_OUTPUT_LEVEL INFO)
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

if (USE_NUMA)
  find_package(NUMA)
endif()

add_library(stencil INTERFACE)
add_library(stencil::stencil ALIAS stencil)

if (STENCIL_OUTPUT_LEVEL MATCHES SPEW)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=0)
elseif (STENCIL_OUTPUT_LEVEL MATCHES DEBUG)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=1)
elseif (STENCIL_OUTPUT_LEVEL MATCHES INFO)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=2)
elseif (STENCIL_OUTPUT_LEVEL MATCHES WARN)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=3)
elseif (STENCIL_OUTPUT_LEVEL MATCHES ERROR)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=4)
elseif (STENCIL_OUTPUT_LEVEL MATCHES FATAL)
  target_compile_definitions(stencil INTERFACE STENCIL_OUTPUT_LEVEL=5)
else()
  message(FATAL_ERROR "unrecognized value of STENCIL_OUTPUT_LEVEL: "  ${STENCIL_OUTPUT_LEVEL}  ", must be one of SPEW DEBUG INFO WARN ERROR FATAL")
endif()

target_compile_features(stencil INTERFACE cxx_std_11)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    stencil
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
     -G
    >
  )
  target_compile_options(
    stencil
    INTERFACE
    -fno-omit-frame-pointer
    -O0
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    stencil
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()

if (USE_CUDA_AWARE_MPI)
  message(STATUS "Using CUDA-Aware MPI")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_CUDA_AWARE_MPI=1)
else()
	message(STATUS "NOT using CUDA-Aware MPI (USE_CUDA_AWARE_MPI=ON to enable)")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_CUDA_AWARE_MPI=0)
endif()

if (MEASURE_TIMINGS)
  message(STATUS "MEASURE_TIMINGS=ON, compiling with -DSTENCIL_MEASURE_TIME=1")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_MEASURE_TIME=1)
else()
  message(STATUS "MEASURE_TIMINGS=OFF, compiling with -DSTENCIL_MEASURE_TIME=0")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_MEASURE_TIME=0)
endif()

if (LOUD)
  message(STATUS "LOUD=ON, compiling with -DSTENCIL_LOUD=1")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_LOUD=1)
else()
  message(STATUS "LOUD=OFF, compiling with -DSTENCIL_LOUD=0")
  target_compile_definitions(stencil INTERFACE -DSTENCIL_LOUD=0)
endif()

if (NUMA_FOUND)
  target_link_libraries(stencil INTERFACE NUMA::NUMA)
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_NUMA=1)
  message(STATUS "NUMA found, compiling with STENCIL_USE_NUMA=1")
else()
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_NUMA=0)
  message(STATUS "NUMA not found, compiling with STENCIL_USE_NUMA=0")
endif()

if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_CUDA=1)
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
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_CUDA=0)
  message(STATUS "CUDA not found, compiling with STENCIL_USE_CUDA=0")
endif()

## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
## causes device linking with -pthread, even as of 3.15.0-rc1
## https://gitlab.kitware.com/cmake/cmake/issues/18897
## Add MPI flags
if(MPI_FOUND)
  target_include_directories(stencil SYSTEM INTERFACE ${MPI_CXX_INCLUDE_DIRS})

  target_link_libraries(stencil INTERFACE ${MPI_CXX_LIBRARIES})
  # target_link_libraries(stencil INTERFACE MPI::MPI_CXX)
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_MPI=1)
  message(STATUS "MPI found, compiling with STENCIL_USE_MPI=1")
  message(STATUS "MPI_CXX_LIBRARIES=" ${MPI_CXX_LIBRARIES})
  message(STATUS "MPIEXEC_EXECUTABLE=" ${MPIEXEC_EXECUTABLE})
else()
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_MPI=0)
  message(STATUS "MPI not found,  compiling with STENCIL_USE_MPI=0")
endif()

## Add include directories
target_include_directories(stencil SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(stencil INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(stencil INTERFACE nvidia-ml)
target_link_libraries(stencil INTERFACE ${NVTOOLSEXT_LIB})

# "this command should be in the source directory root for CTest to find the test file"
enable_testing() 
add_subdirectory(test)
add_subdirectory(src)

