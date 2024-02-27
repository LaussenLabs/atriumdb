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
from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.block import create_gap_arr, create_gap_arr_fast, convert_gap_array_to_intervals, \
    convert_intervals_to_gap_array
from atriumdb.binary_signal_converter import indices_to_signal, signal_to_indices

# Time Types
from atriumdb.block_wrapper import T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES, \
    T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, T_TYPE_START_TIME_NUM_SAMPLES, V_TYPE_INT64, V_TYPE_DOUBLE, \
    V_TYPE_DELTA_INT64, V_TYPE_XOR_DOUBLE

# Window Config
from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.windowing.defintion_splitter import partition_dataset
from atriumdb.windowing.window_config import WindowConfig

# Transfer
from atriumdb.transfer.adb.dataset import transfer_data
