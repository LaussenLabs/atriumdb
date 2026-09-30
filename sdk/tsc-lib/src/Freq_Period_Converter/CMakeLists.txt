add_library(Freq_Period_Converter include/freq_period_converter.h src/freq_to_period.c src/freq_period_converter_p.h src/period_to_freq.c)

target_include_directories(Freq_Period_Converter PUBLIC include)
target_include_directories(Freq_Period_Converter PRIVATE src)

