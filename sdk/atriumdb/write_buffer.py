from atriumdb.adb_functions import time_unit_options


class WriteBuffer:
    def __init__(self, sdk, measure_id, device_id, max_values_buffered=None, gap_tolerance=0, time_units=None):
        self.sdk = sdk
        self.measure_id = measure_id
        self.device_id = device_id
        self.max_values_buffered = max_values_buffered
        time_units = time_units if time_units is not None else 's'
        self.gap_tolerance_nano = int(gap_tolerance * time_unit_options[time_units])
        self.buffered_messages = []
        self.buffered_time_value_pairs = []
        self.total_values_buffered = 0

    def __enter__(self):
        self.sdk._buffer[(self.measure_id, self.device_id)] = self  # Enable buffer in the SDK
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        self.sdk._buffer[(self.measure_id, self.device_id)] = None  # Disable buffer in the SDK

    def push_messages(self, message_list):
        for message in message_list:
            self.total_values_buffered += message['values'].size
        self.buffered_messages.extend(message_list)

        if self.max_values_buffered is not None and self.total_values_buffered >= self.max_values_buffered:
            self.flush()

    def push_time_value_pair(self, data_dict):
        self.total_values_buffered += data_dict['values'].size
        self.buffered_time_value_pairs.append(data_dict)

        if self.max_values_buffered is not None and self.total_values_buffered >= self.max_values_buffered:
            self.flush()

    def flush(self):
        if len(self.buffered_messages) > 0:
            self.sdk._write_messages_to_dataset(
                self.measure_id, self.device_id, self.buffered_messages, self.gap_tolerance_nano)

        if len(self.buffered_time_value_pairs) > 0:
            self.sdk._write_time_value_pairs_to_dataset(
                self.measure_id, self.device_id, self.buffered_time_value_pairs, self.gap_tolerance_nano)

        # Clear the buffer after flushing
        self.buffered_messages.clear()
        self.buffered_time_value_pairs.clear()
        self.total_values_buffered = 0
