add_executable(main main.cu)
target_link_libraries(main stencil::stencil)

add_executable(astaroth-sim astaroth_sim.cu)
target_include_directories(astaroth-sim SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(astaroth-sim stencil::stencil)

add_executable(weak weak.cu)
target_link_libraries(weak stencil::stencil)
target_compile_definitions(weak PRIVATE -DWEAK_METHOD_ALL)

add_executable(weak-staged weak.cu)
target_link_libraries(weak-staged stencil::stencil)

add_executable(weak-colo weak.cu)
target_link_libraries(weak-colo stencil::stencil)
target_compile_definitions(weak-colo PRIVATE -DWEAK_METHOD_COLO)

add_executable(weak-peer weak.cu)
target_link_libraries(weak-peer stencil::stencil)
target_compile_definitions(weak-peer PRIVATE -DWEAK_METHOD_COLO -DWEAK_METHOD_PEER)

add_executable(strong strong.cu)
target_link_libraries(strong stencil::stencil)
