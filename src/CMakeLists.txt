add_executable(main main.cu)
target_link_libraries(main stencil::stencil)

add_executable(astaroth-sim astaroth_sim.cu)
target_include_directories(astaroth-sim SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(astaroth-sim stencil::stencil)

add_executable(weak weak.cu)
target_include_directories(weak SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(weak stencil::stencil)

if (CMAKE_BUILD_TYPE MATCHES Debug)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
target_compile_definitions(
    weak -DNDEBUG
  )
endif()