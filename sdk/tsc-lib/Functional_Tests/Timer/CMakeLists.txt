add_library(Timer include/timer.h src/timer.c)

target_include_directories(Timer PUBLIC include)
target_include_directories(Timer PRIVATE src)