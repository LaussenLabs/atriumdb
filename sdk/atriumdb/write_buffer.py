import logging
import time

_LOGGER = logging.getLogger(__name__)


class WriteBuffer:
    def __init__(self, sdk, max_values_per_measure_device=None, max_total_values_buffered=None,
                 continuous=False, merge_blocks=True):
        self.sdk = sdk
        self.max_values_per_measure_device = max_values_per_measure_device \
            if max_values_per_measure_device is not None else sdk.block.block_size * 100

        self.max_total_values_buffered = max_total_values_buffered \
            if max_total_values_buffered is not None else sdk.block.block_size * 10_000

        # When True, every flushed batch is treated as a single continuous interval.
        self.continuous = continuous

        # When True (the default), small flushes are merged with the closest
        # existing block. A single push with merge_blocks=False disables merging
        # for that sub-buffer's flush.
        self.merge_blocks = merge_blocks

        self.sub_buffers = {}  # Key: (measure_id, device_id), Value: sub-buffer dict
        self.total_values_buffered = 0

    def __enter__(self):
        self.sdk._active_buffer = self  # Set active buffer in the SDK
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Detach the buffer even when the flush raises, otherwise the SDK is left
        # pointing at a dead buffer and every later write silently queues into it.
        try:
            self.flush_all()
        finally:
            self.sdk._active_buffer = None  # Remove active buffer in the SDK

    def _get_sub_buffer(self, key):
        if key not in self.sub_buffers:
            self.sub_buffers[key] = {
                'buffered_messages': [],
                'buffered_time_value_pairs': [],
                'total_values_buffered': 0,
                'last_pushed_time': time.time(),
                'continuous': self.continuous,
                'merge_blocks': self.merge_blocks,
                'gap_tolerance_is_explicit': None,
                'interval_gap_tolerance_nano': None,
            }
        return self.sub_buffers[key]

    @staticmethod
    def _record_gap_tolerance(sub_buffer, gap_tolerance_is_explicit, interval_gap_tolerance_nano):
        """Keep one interval-index policy per measure/device flush.

        A buffered flush merges its pushes into one write, so silently choosing
        one of two policies would make a caller's setting ineffective.  ``None``
        is an automatic policy and is intentionally distinct from explicit zero.
        """
        current = (sub_buffer['gap_tolerance_is_explicit'], sub_buffer['interval_gap_tolerance_nano'])
        requested = (gap_tolerance_is_explicit, interval_gap_tolerance_nano)
        if current[0] is None:
            sub_buffer['gap_tolerance_is_explicit'] = gap_tolerance_is_explicit
            sub_buffer['interval_gap_tolerance_nano'] = interval_gap_tolerance_nano
        elif current != requested:
            raise ValueError(
                "Buffered writes for one measure_id/device_id must use the same gap_tolerance. "
                "Flush the buffer before changing the interval-index policy.")

    def push_segments(self, measure_id, device_id, message_list, continuous=False, merge_blocks=True):
        key = (measure_id, device_id)
        sub_buffer = self._get_sub_buffer(key)
        first = message_list[0]
        self._record_gap_tolerance(
            sub_buffer, first['gap_tolerance_is_explicit'], first['interval_gap_tolerance_nano'])
        sub_buffer['buffered_messages'].extend(message_list)
        sub_buffer['continuous'] = sub_buffer['continuous'] or continuous
        sub_buffer['merge_blocks'] = sub_buffer['merge_blocks'] and merge_blocks
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

    def push_time_value_pairs(self, measure_id, device_id, data_dict, continuous=False, merge_blocks=True):
        key = (measure_id, device_id)
        sub_buffer = self._get_sub_buffer(key)
        self._record_gap_tolerance(
            sub_buffer, data_dict['gap_tolerance_is_explicit'], data_dict['interval_gap_tolerance_nano'])
        sub_buffer['buffered_time_value_pairs'].append(data_dict)
        sub_buffer['continuous'] = sub_buffer['continuous'] or continuous
        sub_buffer['merge_blocks'] = sub_buffer['merge_blocks'] and merge_blocks
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
        continuous = sub_buffer['continuous']
        merge_blocks = sub_buffer['merge_blocks']
        # The sub-buffer is dropped whether or not the write succeeds: a batch that
        # the write path rejected will be rejected identically on every retry, and
        # leaving it queued would make a later flush of an unrelated measure fail too.
        try:
            if sub_buffer['buffered_messages']:
                self.sdk._write_segments_to_dataset(
                    measure_id, device_id, sub_buffer['buffered_messages'],
                    interval_gap_tolerance_nano=sub_buffer['interval_gap_tolerance_nano'], continuous=continuous,
                    merge_blocks=merge_blocks)

            if sub_buffer['buffered_time_value_pairs']:
                self.sdk._write_time_value_pairs_to_dataset(
                    measure_id, device_id, sub_buffer['buffered_time_value_pairs'],
                    interval_gap_tolerance_nano=sub_buffer['interval_gap_tolerance_nano'], continuous=continuous,
                    merge_blocks=merge_blocks)
        finally:
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
        """Flush every sub-buffer, isolating failures per (measure, device).

        One rejected batch must not discard the other measures queued in the same
        context: every sub-buffer is attempted, and the first failure is
        re-raised afterwards so the caller still sees the error.
        """
        first_error = None
        failed_keys = []
        for key in list(self.sub_buffers.keys()):
            try:
                self.flush_sub_buffer(key)
            except Exception as flush_error:
                failed_keys.append(key)
                if first_error is None:
                    first_error = flush_error

        if first_error is not None:
            _LOGGER.error(
                f"{len(failed_keys)} buffered (measure_id, device_id) batches failed to flush "
                f"{failed_keys}; every other buffered batch was written. Re-raising the first "
                f"failure.")
            raise first_error
