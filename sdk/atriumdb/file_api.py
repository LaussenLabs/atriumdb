from pathlib import Path
import uuid
import numpy as np


class AtriumFileHandler:
    def __init__(self, top_level_dir):
        self.top_level_dir = top_level_dir

    def generate_tsc_filename(self, measure_id, device_id):
        hex_str = uuid.uuid4().hex
        directory = "{}/{}/{}".format(self.top_level_dir, device_id, measure_id)
        Path(directory).mkdir(parents=True, exist_ok=True)
        return "{}.tsc".format(hex_str)

    def to_abs_path(self, filename, measure_id, device_id):
        return "{}/{}/{}/{}".format(self.top_level_dir, device_id, measure_id, filename)

    def write_bytes(self, measure_id, device_id, encoded_buffer):
        filename = self.generate_tsc_filename(measure_id, device_id)
        with open(self.to_abs_path(filename, measure_id, device_id), 'wb') as file:
            file.write(encoded_buffer)
        return filename

    def read_bytes(self, measure_id, device_id, filename, start_byte, num_bytes):
        if not Path(filename).is_file():
            filename = self.to_abs_path(filename, measure_id, device_id)
        with open(filename, 'rb') as file:
            file.seek(start_byte)
            return np.frombuffer(file.read(num_bytes), dtype=np.uint8)

    def read_bytes_fast(self, measure_id, device_id, filename, start_byte, num_bytes):
        with open(self.to_abs_path(filename, measure_id, device_id), 'rb') as file:
            file.seek(start_byte)
            return file.read(num_bytes)

    def read_into_bytearray(self, measure_id, device_id, filename, start_byte, buffer):
        with open(self.to_abs_path(filename, measure_id, device_id), 'rb') as file:
            file.seek(start_byte)
            file.readinto(buffer)

    def read_file_list_1(self, measure_id, read_list, filename_dict):
        return np.concatenate(
                [self.read_bytes(measure_id, block_device_id, filename_dict[file_id], start_byte, num_bytes)
                 for block_device_id, file_id, start_byte, num_bytes in read_list], axis=None)

    def read_file_list_2(self, measure_id, read_list, filename_dict):
        all_bytes = b''.join([
            self.read_bytes_fast(measure_id, block_device_id, filename_dict[file_id], start_byte, num_bytes)
            for block_device_id, file_id, start_byte, num_bytes in read_list])

        return np.frombuffer(all_bytes, dtype=np.uint8)

    def read_file_list_3(self, measure_id, read_list, filename_dict):
        all_bytes = bytearray(sum([num_bytes for _, _, _, num_bytes in read_list]))
        mem_view = memoryview(all_bytes)

        index = 0
        for block_device_id, file_id, start_byte, num_bytes in read_list:
            self.read_into_bytearray(measure_id, block_device_id, filename_dict[file_id],
                                     start_byte, mem_view[index:index+num_bytes])
            index += num_bytes

        return np.frombuffer(all_bytes, dtype=np.uint8)
