add_executable(test_all test_main.cu test_pack.cu test_any_tx.cu)
target_include_directories(test_all SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(test_all stencil)
add_test(NAME test_all COMMAND test_all -a)