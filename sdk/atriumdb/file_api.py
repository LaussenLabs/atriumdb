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

from pathlib import Path
import uuid
import numpy as np


class AtriumFileHandler:
    def __init__(self, top_level_dir):
        self.top_level_dir = top_level_dir

    def generate_tsc_filename(self, measure_id, device_id):
        # Generate a random UUID and convert it to a hexadecimal string
        hex_str = uuid.uuid4().hex

        # Create the parent directory.
        directory = "{}/{}/{}".format(self.top_level_dir, device_id, measure_id)
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Return the unique filename with a .tsc extension
        return "{}.tsc".format(hex_str)

    def to_abs_path(self, filename, measure_id, device_id):
        # Return the absolute path of the file in the directory structure
        return "{}/{}/{}/{}".format(self.top_level_dir, device_id, measure_id, filename)

    def write_bytes(self, measure_id, device_id, encoded_buffer):
        # Generate a unique filename using the generate_tsc_filename function
        filename = self.generate_tsc_filename(measure_id, device_id)

        # Get the absolute path of the file using the to_abs_path function
        abs_path = self.to_abs_path(filename, measure_id, device_id)

        # Write the binary data (encoded_buffer) to the file
        with open(abs_path, 'wb') as file:
            file.write(encoded_buffer)

        # Return the generated filename
        return filename

    def read_bytes(self, measure_id, device_id, filename, start_byte, num_bytes):
        # Check if the file exists as is.
        if not Path(filename).is_file():
            # If not, get the absolute path of the file.
            filename = self.to_abs_path(filename, measure_id, device_id)

        # Open the file in binary mode
        with open(filename, 'rb') as file:
            # Seek to the specified start byte position
            file.seek(start_byte)
            # Read the specified number of bytes and return the data as a NumPy array of unsigned 8-bit integers
            return np.frombuffer(file.read(num_bytes), dtype=np.uint8)

    def read_bytes_fast(self, measure_id, device_id, filename, start_byte, num_bytes):
        # Open the file in binary mode with the absolute path provided by the to_abs_path function
        with open(self.to_abs_path(filename, measure_id, device_id), 'rb') as file:
            # Seek to the specified start byte position
            file.seek(start_byte)
            # Read the specified number of bytes and return the data as a byte string
            return file.read(num_bytes)

    def read_into_bytearray(self, measure_id, device_id, filename, start_byte, buffer):
        # Open the file in binary mode with the absolute path provided by the to_abs_path function
        with open(self.to_abs_path(filename, measure_id, device_id), 'rb') as file:
            # Seek to the specified start byte position
            file.seek(start_byte)
            # Read the data into the given bytearray object
            file.readinto(buffer)

    def read_file_list_1(self, measure_id, read_list, filename_dict):
        # Read specified bytes from files using read_bytes and concatenate the results into a NumPy array
        return np.concatenate(
            [self.read_bytes(measure_id, block_device_id, filename_dict[file_id], start_byte, num_bytes)
             for block_device_id, file_id, start_byte, num_bytes in read_list], axis=None)

    def read_file_list_2(self, measure_id, read_list, filename_dict):
        # Read specified bytes from files using read_bytes_fast and concatenate the results using join
        all_bytes = b''.join([
            self.read_bytes_fast(measure_id, block_device_id, filename_dict[file_id], start_byte, num_bytes)
            for block_device_id, file_id, start_byte, num_bytes in read_list])

        # Create a NumPy array from the concatenated bytes using np.frombuffer
        return np.frombuffer(all_bytes, dtype=np.uint8)

    def read_file_list_3(self, measure_id, read_list, filename_dict):
        # Create a bytearray with the total size of the data to be read
        all_bytes = bytearray(sum([num_bytes for _, _, _, num_bytes in read_list]))
        # Create a memoryview of the bytearray
        mem_view = memoryview(all_bytes)

        # Initialize an index to keep track of the current position in the memoryview
        index = 0
        # Iterate through the read_list and read the specified bytes from the files using read_into_bytearray
        for block_device_id, file_id, start_byte, num_bytes in read_list:
            self.read_into_bytearray(measure_id, block_device_id, filename_dict[file_id],
                                     start_byte, mem_view[index:index + num_bytes])
            # Update the index for the next iteration
            index += num_bytes

        # Create a NumPy array from the bytearray using np.frombuffer
        return np.frombuffer(all_bytes, dtype=np.uint8)
