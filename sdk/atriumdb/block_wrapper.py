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

from ctypes import *
import numpy as np
import platform
from pathlib import Path
import os

from typing import Tuple

# Current Block Data
TSC_VERSION_NUM = 2
TSC_VERSION_EXT = 3
TSC_NUM_CHANNELS = 1

# Time Types
T_TYPE_TIMESTAMP_ARRAY_INT64_NANO = 1
T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO = 2
T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES = 3
T_TYPE_START_TIME_NUM_SAMPLES = 4

# Value Types
V_TYPE_INT64 = 1
V_TYPE_DOUBLE = 2

V_TYPE_DELTA_INT64 = 3
V_TYPE_XOR_DOUBLE = 4


class BlockMetadata(Structure):
    # _pack_ = 1  # Allows the structure to be interpreted through bytes.
    _fields_ = [("tsc_version_num", c_uint8),
                ("tsc_version_ext", c_uint8),
                ("num_vals", c_uint64),
                ("num_channels", c_uint8),
                ("meta_num_bytes", c_uint16),

                # Time Metadata
                ("t_raw_type", c_uint8),
                ("t_raw_size", c_uint64),
                ("t_encoded_type", c_uint8),
                ("t_encoded_size", c_uint64),
                ("t_compression", c_uint8),
                ("t_compression_level", c_int8),
                ("start_n", c_uint64),
                ("end_n", c_uint64),
                ("freq_nhz", c_uint64),
                ("num_gaps", c_uint64),
                ("t_start_byte", c_uint64),
                ("t_num_bytes", c_uint64),

                # Value Metadata
                ("v_raw_type", c_uint8),
                ("v_raw_size", c_uint64),
                ("v_encoded_type", c_uint8),
                ("v_encoded_size", c_uint64),
                ("v_compression", c_uint8),
                ("v_compression_level", c_int8),
                ("bytes_per_value", c_uint8),
                ("order", c_uint8),
                ("max", c_double),
                ("min", c_double),
                ("mean", c_double),
                ("std", c_double),
                ("scale_m", c_double),
                ("scale_b", c_double),
                ("v_start_byte", c_uint64),
                ("v_num_bytes", c_uint64)]


class BlockMetadataWrapper:
    def __init__(self, block_metadata):
        self.tsc_version_num = block_metadata.tsc_version_num
        self.tsc_version_ext = block_metadata.tsc_version_ext
        self.num_vals = block_metadata.num_vals
        self.num_channels = block_metadata.num_channels
        self.meta_num_bytes = block_metadata.meta_num_bytes

        self.t_raw_type = block_metadata.t_raw_type
        self.t_raw_size = block_metadata.t_raw_size
        self.t_encoded_type = block_metadata.t_encoded_type
        self.t_encoded_size = block_metadata.t_encoded_size
        self.t_compression = block_metadata.t_compression
        self.t_compression_level = block_metadata.t_compression_level

        self.start_n = block_metadata.start_n
        self.end_n = block_metadata.end_n
        self.freq_nhz = block_metadata.freq_nhz
        self.num_gaps = block_metadata.num_gaps
        self.t_start_byte = block_metadata.t_start_byte
        self.t_num_bytes = block_metadata.t_num_bytes

        self.v_raw_type = block_metadata.v_raw_type
        self.v_raw_size = block_metadata.v_raw_size
        self.v_encoded_type = block_metadata.v_encoded_type
        self.v_encoded_size = block_metadata.v_encoded_size
        self.v_compression = block_metadata.v_compression
        self.v_compression_level = block_metadata.v_compression_level
        self.bytes_per_value = block_metadata.bytes_per_value
        self.order = block_metadata.order
        self.max = block_metadata.max
        self.min = block_metadata.min
        self.mean = block_metadata.mean
        self.std = block_metadata.std
        self.scale_m = block_metadata.scale_m
        self.scale_b = block_metadata.scale_b
        self.v_start_byte = block_metadata.v_start_byte
        self.v_num_bytes = block_metadata.v_num_bytes


class BlockOptions(Structure):
    _fields_ = [("delta_order_min", c_uint8),
                ("delta_order_max", c_uint8),
                ("bytes_per_value_min", c_uint8)]


class WrappedBlockDll:

    def __init__(self, abs_path_to_dll: str, num_threads: int):
        self.num_threads = num_threads
        if platform.system() == "Windows":
            # Windows
            os.add_dll_directory(Path(abs_path_to_dll).parent)
            self.bc_dll = WinDLL(abs_path_to_dll)
        elif platform.system() == "Linux":
            # Linux
            self.bc_dll = CDLL(abs_path_to_dll)

        else:
            # MACOS
            self.bc_dll = CDLL(abs_path_to_dll)

        self.bc_dll.block_get_buffer_size.argtypes = [c_void_p, c_uint64, POINTER(c_uint64), POINTER(c_uint64),
                                                            POINTER(BlockMetadata)]
        self.bc_dll.block_get_buffer_size.restype = c_size_t

        self.bc_dll.encode_blocks.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64, POINTER(c_uint64),
                                              POINTER(c_uint64), POINTER(c_uint64), POINTER(BlockMetadata),
                                              POINTER(BlockOptions), c_uint16]
        self.bc_dll.encode_blocks.restype = c_size_t

        self.bc_dll.decode_blocks.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64, POINTER(c_uint64),
                                              POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), c_uint16]

        # Set up 'convert_value_data_to_analog' method
        self.bc_dll.convert_value_data_to_analog.argtypes = [
            c_void_p,                # void *value_data
            POINTER(c_double),       # double *analog_values
            POINTER(BlockMetadata),  # block_metadata_t *headers
            POINTER(c_uint64),       # uint64 *analog_values
            c_uint64                 # uint64_t num_blocks
        ]
        self.bc_dll.convert_value_data_to_analog.restype = None

    def encode_blocks_sdk(self, time_data: np.ndarray, value_data: np.ndarray, num_blocks: int,
                          t_block_start: np.ndarray, v_block_start: np.ndarray, headers: POINTER(BlockMetadata),
                          options: BlockOptions) -> Tuple[np.ndarray, np.ndarray]:
        byte_start = np.zeros(num_blocks, dtype=np.uint64)
        time_data.ctypes.data_as(c_void_p), num_blocks, t_block_start.ctypes.data_as(POINTER(c_uint64)),
        byte_start.ctypes.data_as(POINTER(c_uint64)), headers

        buffer_size = self.bc_dll.block_get_buffer_size(
            time_data.ctypes.data_as(c_void_p), num_blocks, t_block_start.ctypes.data_as(POINTER(c_uint64)),
            byte_start.ctypes.data_as(POINTER(c_uint64)), headers)

        encoded_bytes = np.zeros(buffer_size, dtype=np.uint8)
        len_encoded = self.bc_dll.encode_blocks(
            time_data.ctypes.data_as(c_void_p), value_data.ctypes.data_as(c_void_p),
            encoded_bytes.ctypes.data_as(c_void_p),
            num_blocks, t_block_start.ctypes.data_as(POINTER(c_uint64)),
            v_block_start.ctypes.data_as(POINTER(c_uint64)),
            byte_start.ctypes.data_as(POINTER(c_uint64)), headers, byref(options), self.num_threads)

        return encoded_bytes[:len_encoded], byte_start

    def decode_blocks_sdk(self, time_data: np.ndarray, value_data: np.ndarray, encoded_bytes: np.ndarray,
                          num_blocks: int, t_block_start: np.ndarray, v_block_start: np.ndarray,
                          byte_start: np.ndarray, t_byte_start: np.ndarray):
        self.bc_dll.decode_blocks(
            time_data.ctypes.data_as(c_void_p), value_data.ctypes.data_as(c_void_p),
            encoded_bytes.ctypes.data_as(c_void_p),
            num_blocks, t_block_start.ctypes.data_as(POINTER(c_uint64)),
            v_block_start.ctypes.data_as(POINTER(c_uint64)),
            byte_start.ctypes.data_as(POINTER(c_uint64)), t_byte_start.ctypes.data_as(POINTER(c_uint64)),
            self.num_threads)

    def convert_value_data_to_analog(self, value_data: np.ndarray, analog_values: np.ndarray,
                                     headers: list[BlockMetadata], num_blocks: int):
        # Define the array type for BlockMetadata
        BlockMetadataArray = BlockMetadata * num_blocks

        # Create an array of BlockMetadata
        headers_array = BlockMetadataArray(*headers)

        analog_block_start_index_array = np.zeros(num_blocks, dtype=np.uint64)
        total_num_values = 0
        for i, h in enumerate(headers):
            analog_block_start_index_array[i] = total_num_values
            total_num_values += h.num_vals

        self.bc_dll.convert_value_data_to_analog(
            value_data.ctypes.data_as(c_void_p),  # value_data as void pointer
            analog_values.ctypes.data_as(POINTER(c_double)),  # analog_values as pointer to double
            cast(headers_array, POINTER(BlockMetadata)),  # headers as pointer to BlockMetadata array
            analog_block_start_index_array.ctypes.data_as(POINTER(c_uint64)),  # block_start_array as pointer to uint64
            c_uint64(num_blocks)  # num_blocks as uint64
        )
