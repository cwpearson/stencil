# - Try to find libnuma
# Once done, this will define
#
#  NUMA_FOUND - system has NUMA
#  NUMA_INCLUDE_DIRS - the NUMA include directories
#  NUMA_LIBRARIES - link these to use NUMA
#
# It will also create an imported target NUMA::NUMA
# 



SET(NUMA_INCLUDE_SEARCH_PATHS
      ${NUMA}
      /usr/include
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/include
)

SET(NUMA_LIBRARY_SEARCH_PATHS
      ${NUMA}
      /usr/lib
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/lib
)

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h
  PATHS ${NUMA_INCLUDE_SEARCH_PATHS}
  DOC "NUMA include directory")

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${NUMA_LIBRARY_SEARCH_PATHS}
  DOC "NUMA library")

if (numa_LIBRARY)
    get_filename_component(NUMA_LIBRARY_DIR ${NUMA_LIBRARY} PATH)
endif()


if (NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
  if (NOT TARGET NUMA::NUMA)
    add_library(NUMA::NUMA SHARED IMPORTED)
    set_target_properties(NUMA::NUMA PROPERTIES
      IMPORTED_LOCATION "${NUMA_LIBRARY}"
    )
    target_include_directories(NUMA::NUMA INTERFACE "${NUMA_INCLUDE_DIR}")
    target_link_libraries(NUMA::NUMA INTERFACE "${NUMA_LIBRARY}")
  else()
    message(WARNING "NUMA::NUMA is already a target")
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA 
  DEFAULT_MESSAGE
  NUMA_LIBRARY NUMA_INCLUDE_DIR
)

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY_DIR NUMA_LIBRARY)

# message(STATUS "NUMA_FOUND: " ${NUMA_FOUND})
# message(STATUS "NUMA_LIBRARY: " ${NUMA_LIBRARY})
# message(STATUS "NUMA_INCLUDE_DIR: " ${NUMA_INCLUDE_DIR})
# get_property(propval TARGET NUMA::NUMA PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "INTERFACE_INCLUDE_DIRECTORIES: " ${propval})
# get_property(propval TARGET NUMA::NUMA PROPERTY INTERFACE_LINK_LIBRARIES)
# message(STATUS "INTERFACE_LINK_LIBRARIES: " ${propval})