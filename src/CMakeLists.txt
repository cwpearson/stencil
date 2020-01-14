add_executable(main main.cu)
target_link_libraries(main stencil::stencil)

add_executable(astaroth-sim astaroth_sim.cu)
target_link_libraries(astaroth-sim stencil::stencil)