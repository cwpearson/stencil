set(STENCIL_SOURCES ${STENCIL_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/copy.cu
  ${CMAKE_CURRENT_LIST_DIR}/gpu_topology.cpp
  ${CMAKE_CURRENT_LIST_DIR}/local_domain.cu
  ${CMAKE_CURRENT_LIST_DIR}/machine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/numeric.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pack_kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/packer.cu
  ${CMAKE_CURRENT_LIST_DIR}/rcstream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/stencil.cu
  ${CMAKE_CURRENT_LIST_DIR}/timer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/translate.cu
  ${CMAKE_CURRENT_LIST_DIR}/tx_colocated_direct_access.cu
  ${CMAKE_CURRENT_LIST_DIR}/tx_ipc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/unpack_kernel.cu
)

set(STENCIL_SOURCES 
  ${STENCIL_SOURCES}
  PARENT_SCOPE
)