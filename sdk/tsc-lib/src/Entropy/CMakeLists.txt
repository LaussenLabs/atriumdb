add_library(Entropy include/entropy.h src/entropy.c src/entropy_buffer.c src/entropy_p.h src/entropy_d64.c)

target_include_directories(Entropy PUBLIC include)
target_include_directories(Entropy PRIVATE src)

target_link_libraries(Entropy PRIVATE m Distinct)
