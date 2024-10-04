import time

from atriumdb.adb_functions import time_unit_options


class WriteBuffer:
    def __init__(self, sdk, max_values_per_measure_device=None, max_total_values_buffered=None, gap_tolerance=0, time_units=None):
        self.sdk = sdk
        self.max_values_per_measure_device = max_values_per_measure_device \
            if max_values_per_measure_device is not None else sdk.block.block_size * 100

        self.max_total_values_buffered = max_total_values_buffered \
            if max_total_values_buffered is not None else sdk.block.block_size * 10_000

        if gap_tolerance > 0:
            if time_units is None:
                raise ValueError('If you are using a non-zero gap_tolerance you must specify time_units, '
                                 'one of ["s", "ms", "us", "ns"]')
            self.gap_tolerance_nano = int(gap_tolerance * time_unit_options[time_units])
        else:
            self.gap_tolerance_nano = 0

        self.sub_buffers = {}  # Key: (measure_id, device_id), Value: sub-buffer dict
        self.total_values_buffered = 0

    def __enter__(self):
        self.sdk._active_buffer = self  # Set active buffer in the SDK
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush_all()
        self.sdk._active_buffer = None  # Remove active buffer in the SDK

    def push_segments(self, measure_id, device_id, message_list):
        key = (measure_id, device_id)
        if key not in self.sub_buffers:
            self.sub_buffers[key] = {
                'buffered_messages': [],
                'buffered_time_value_pairs': [],
                'total_values_buffered': 0,
                'last_pushed_time': time.time()
            }
        sub_buffer = self.sub_buffers[key]
        sub_buffer['buffered_messages'].extend(message_list)
        num_values = sum(m['values'].size for m in message_list)
        sub_buffer['total_values_buffered'] += num_values
        self.total_values_buffered += num_values
        sub_buffer['last_pushed_time'] = time.time()

        # Check if sub-buffer exceeds max_values_per_measure_device
        if self.max_values_per_measure_device is not None and sub_buffer['total_values_buffered'] >= self.max_values_per_measure_device:
            # Flush this sub-buffer
            self.flush_sub_buffer(key)

        # Check if total_values_buffered exceeds max_total_values_buffered
        if self.max_total_values_buffered is not None and self.total_values_buffered >= self.max_total_values_buffered:
            # Flush oldest sub-buffer that has values
            self.flush_oldest_sub_buffer()

    def push_time_value_pairs(self, measure_id, device_id, data_dict):
        key = (measure_id, device_id)
        if key not in self.sub_buffers:
            self.sub_buffers[key] = {
                'buffered_messages': [],
                'buffered_time_value_pairs': [],
                'total_values_buffered': 0,
                'last_pushed_time': time.time()
            }
        sub_buffer = self.sub_buffers[key]
        sub_buffer['buffered_time_value_pairs'].append(data_dict)
        num_values = data_dict['values'].size
        sub_buffer['total_values_buffered'] += num_values
        self.total_values_buffered += num_values
        sub_buffer['last_pushed_time'] = time.time()

        # Check if sub-buffer exceeds max_values_per_measure_device
        if sub_buffer['total_values_buffered'] >= self.max_values_per_measure_device:
            # Flush this sub-buffer
            self.flush_sub_buffer(key)

        # Check if total_values_buffered exceeds max_total_values_buffered
        if self.total_values_buffered >= self.max_total_values_buffered:
            # Flush oldest sub-buffer that has values
            self.flush_oldest_sub_buffer()

    def flush_sub_buffer(self, key):
        sub_buffer = self.sub_buffers.get(key)
        if sub_buffer is None:
            return

        measure_id, device_id = key
        if sub_buffer['buffered_messages']:
            self.sdk._write_segments_to_dataset(
                measure_id, device_id, sub_buffer['buffered_messages'], self.gap_tolerance_nano)

        if sub_buffer['buffered_time_value_pairs']:
            self.sdk._write_time_value_pairs_to_dataset(
                measure_id, device_id, sub_buffer['buffered_time_value_pairs'], self.gap_tolerance_nano)


        self.total_values_buffered -= sub_buffer['total_values_buffered']

        del self.sub_buffers[key]

    def flush_oldest_sub_buffer(self):
        # Find the sub-buffer with the oldest last_pushed_time
        oldest_key = None
        oldest_time = None
        for key, sub_buffer in self.sub_buffers.items():
            if sub_buffer['total_values_buffered'] > 0:
                if oldest_time is None or sub_buffer['last_pushed_time'] < oldest_time:
                    oldest_time = sub_buffer['last_pushed_time']
                    oldest_key = key
        if oldest_key is not None:
            self.flush_sub_buffer(oldest_key)

    def flush_all(self):
        for key in list(self.sub_buffers.keys()):
            self.flush_sub_buffer(key)
