add_library(csv_reader include/csv_reader.h src/csv_to_time_value_arr.c src/csv_p.h src/csv_reader.c)

target_include_directories(csv_reader PUBLIC include)
target_include_directories(csv_reader PRIVATE src)