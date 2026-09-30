add_library(Distinct include/distinct.h src/count_distinct_d64.c src/count_arr_d64.c)

target_include_directories(Distinct PUBLIC include)
target_include_directories(Distinct PRIVATE src)