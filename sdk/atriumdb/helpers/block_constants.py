# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

COMPRESSION_TYPES = {'NONE': 1, "ZSTD": 3, "LZ4": 4, "LZ4HC": 5}
COMPRESSION_LEVELS = {'NONE': [0], "ZSTD": list(range(1, 23)), "LZ4": [0], "LZ4HC": list(range(10))}

TIME_TYPES = {'TIME_ARRAY_INT64_NS': 1, 'GAP_ARRAY_INT64_INDEX_DURATION_NS': 2, 'GAP_ARRAY_INT64_INDEX_NUM_SAMPLES': 3,
              'START_TIME_NUM_SAMPLES': 4}
VALUE_TYPES = {'INT64': 1, 'DOUBLE': 2, 'DELTA_INT64': 3, 'XOR_DOUBLE': 4}
