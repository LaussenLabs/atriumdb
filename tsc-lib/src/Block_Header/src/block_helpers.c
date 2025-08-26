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
// Created by Will Dixon on 2025-08-26.
//

#include <block_header.h>
#include <freq_period_converter.h>

uint64_t get_period_ns_from_header(const block_metadata_t *header)
{
    // If version is 2.4 or higher, freq_nhz is actually period_ns
    if (header->tsc_version_num > 2 || (header->tsc_version_num == 2 && header->tsc_version_ext >= 4)) {
        return header->freq_nhz;  // freq_nhz is actually period_ns
    } else {
        return uint64_nhz_freq_to_uint64_ns_period(header->freq_nhz);
    }
}