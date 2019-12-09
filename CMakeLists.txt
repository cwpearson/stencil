# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(stencil LANGUAGES CXX CUDA VERSION 0.1.0.0)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

option(USE_NUMA "compile with NUMA support" ON)
option(USE_MPI "compile with MPI support" ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake")


get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

if(USE_MPI)
find_package(MPI)
endif()

if (USE_NUMA)
find_package(NUMA)
endif()

add_library(stencil INTERFACE)
add_library(stencil::stencil ALIAS stencil)


if (NUMA_FOUND)
  target_link_libraries(stencil INTERFACE NUMA::NUMA)
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_NUMA=1)
  message(STATUS "NUMA found, compiling with STENCIL_USE_NUMA=1")
else()
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_NUMA=0)
  message(STATUS "NUMA not found, compiling with STENCIL_USE_NUMA=0")
endif()

## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
## causes device linking with -pthread, even as of 3.15.0-rc1
## https://gitlab.kitware.com/cmake/cmake/issues/18897
## Add MPI flags
if(MPI_FOUND)
  target_include_directories(stencil SYSTEM INTERFACE ${MPI_CXX_INCLUDE_DIRS})
  message(STATUS ${MPI_CXX_LIBRARIES})
  target_link_libraries(stencil INTERFACE ${MPI_CXX_LIBRARIES})
  # target_link_libraries(stencil INTERFACE MPI::MPI_CXX)
  target_compile_definitions(stencil INTERFACE -DSTENCIL_USE_MPI=1)
  message(STATUS "MPI found, compiling with STENCIL_USE_MPI=1")
else()
  target_compile_definitions(STENCIL INTERFACE -DSTENCIL_USE_MPI=0)
  message(STATUS "MPI found,  compiling with STENCIL_USE_MPI=0")
endif()

## Add include directories
target_include_directories(stencil SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(stencil INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# "this command should be in the source directory root for CTest to find the test file"
enable_testing() 
add_subdirectory(test)
add_subdirectory(src)

