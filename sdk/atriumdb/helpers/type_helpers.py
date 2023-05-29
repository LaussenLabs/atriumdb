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

from atriumdb.helpers.block_constants import *
import numpy as np


def get_compression_levels_help(compression_type):
    if compression_type is None:
        return COMPRESSION_LEVELS

    elif type(compression_type) != str and type(compression_type) != int:
        raise TypeError(
            "Variable compression_type must be a string, int or None, not {}.".format(type(compression_type)))

    elif type(compression_type) == int and compression_type not in COMPRESSION_TYPES.values():
        raise ValueError("Variable compression_type must be one of {}.".format(list(COMPRESSION_TYPES.values())))

    elif compression_type.upper() not in COMPRESSION_LEVELS.keys():
        raise ValueError("Variable compression_type must be one of {}.".format(list(COMPRESSION_TYPES.keys())))

    else:
        if type(compression_type) == int:
            compression_type = dict_inverse(COMPRESSION_TYPES)[compression_type]

        return COMPRESSION_LEVELS[compression_type.upper()]


def t_type_raw_choose(time_data_len: int, value_data_len: int, freq_nhz: int) -> int:
    if time_data_len == value_data_len:
        return TIME_TYPES['TIME_ARRAY_INT64_NS']

    elif (10 ** 18) % freq_nhz == 0:
        return TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']

    else:
        return TIME_TYPES['GAP_ARRAY_INT64_INDEX_NUM_SAMPLES']


def v_type_raw_choose(sample_value) -> int:
    if np.issubdtype(sample_value, np.floating):
        return VALUE_TYPES['DOUBLE']

    elif np.issubdtype(sample_value, np.integer):
        return VALUE_TYPES['INT64']

    else:
        raise TypeError("Values must hold ints or floats, not {}.".format(type(sample_value)))


def t_type_encoded_choose(t_type_raw: int, freq_nhz: int) -> int:
    if t_type_raw != TIME_TYPES['TIME_ARRAY_INT64_NS']:
        return t_type_raw

    elif (10 ** 18) % freq_nhz == 0:
        return TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']

    else:
        return TIME_TYPES['GAP_ARRAY_INT64_INDEX_NUM_SAMPLES']


def v_type_encoded_choose(v_type_raw: int) -> int:
    if v_type_raw == VALUE_TYPES['INT64']:
        return VALUE_TYPES['DELTA_INT64']

    else:
        return VALUE_TYPES['DOUBLE']


def dict_inverse(dictionary):
    return {v: k for k, v in dictionary.items()}
