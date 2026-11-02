add_library(Size include/size.h src/size_encode.c src/size_best.c src/size_decode.c)

target_include_directories(Size PUBLIC include)
target_include_directories(Size PRIVATE src)