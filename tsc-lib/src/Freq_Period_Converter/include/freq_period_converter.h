/*
    AtriumDB is a timeseries database software designed to best handle the unique features and
    challenges that arise from clinical waveform data.
        Copyright (C) 2023  The Hospital for Sick Children

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https:www.gnu.org/licenses/>.
*/

//
// Created by Will Dixon on 2021-05-17.
//

#ifndef MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H
#define MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H

#include <stdint.h>

uint64_t uint64_nhz_freq_to_uint64_ns_period(uint64_t freq_nhz);

uint64_t uint64_ns_period_to_uint64_nhz_freq(uint64_t period_ns);

#endif //MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H
