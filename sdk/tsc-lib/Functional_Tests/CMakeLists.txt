add_subdirectory(Timer)
add_subdirectory(csv_reader)

add_executable(lct large_scale_test.c)

target_link_libraries(lct Timer Block)