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
from collections import Counter


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

    def read_file_list(self, read_list, filename_dict):
        # get number of times each file needs to be read by seeing how many times a file_id appears in the read list
        file_read_counts = Counter([block[2] for block in read_list])

        # Create a bytearray with the total size of the data to be read
        all_bytes = bytearray(sum([num_bytes for _, _, _, _, num_bytes in read_list]))
        # Create a memoryview of the bytearray
        mem_view = memoryview(all_bytes)

        # Initialize an index to keep track of the current position in the memoryview and an open_file_dict to keep
        # track of open files
        open_files, index = {}, 0
        try:
            # Iterate through the read_list and read the specified bytes from the files using read_into_bytearray
            for measure_id, device_id, file_id, start_byte, num_bytes in read_list:

                # if we have not already opened the file open it and add it to the dictionary
                if file_id not in open_files:
                    # Open the file in binary mode with the absolute path provided by the to_abs_path function
                    open_files[file_id] = open(self.to_abs_path(filename_dict[file_id], measure_id, device_id), 'rb')

                # get open file from dictionary
                file = open_files[file_id]
                # subtract 1 from the number of times the file needs to be read from
                file_read_counts[file_id] -= 1

                # Find the specified start byte position in the file
                file.seek(start_byte)
                # Read the data into the given bytearray object
                file.readinto(mem_view[index:index + num_bytes])

                # Update the index for the next iteration
                index += num_bytes

                # check if the file read count is 0 and if it has, close the file and remove it from the dictionary
                if file_read_counts[file_id] == 0:
                    open_files[file_id].close()
                    del open_files[file_id]

        # if there is an error in the program make sure to close all open files
        finally:
            [open_files[file_id].close() for file_id, open_file in open_files.items()]

        # Create a NumPy array from the bytearray using np.frombuffer
        return np.frombuffer(all_bytes, dtype=np.uint8)

    def read_from_filepath(self, file_path):
        # Ensure the file_path is a Path object
        file_path = Path(file_path)

        # Read the file in binary mode
        with file_path.open('rb') as file:
            file_bytes = file.read()

        # Create a writable numpy array from the file bytes
        writable_file_bytes = np.frombuffer(file_bytes, dtype=np.uint8).copy()

        return writable_file_bytes
