import unittest

import numpy as np
import random

from atriumdb.block import create_gap_arr, interpret_gap_arr, Block
from atriumdb.helpers.block_constants import TIME_TYPES, VALUE_TYPES
from generate_data import generate_random_sdk_compliant_data


class TestBlockCodec(unittest.TestCase):
    def test_encode_decode(self):
        num_messages = 10 ** 2

        bc = Block("../data/libBlock_Codec.dylib")

        for _ in range(10):
            if random.random() < 0.333:
                samples_per_message = 1
            else:
                samples_per_message = None

            value_data, nom_times, samples_per_message, freq_nhz = \
                generate_random_sdk_compliant_data(num_messages, samples_per_message=samples_per_message)

            gap_arr = create_gap_arr(nom_times, samples_per_message, freq_nhz)

            if samples_per_message == 1:
                values = value_data
            else:
                values_2d, num_samples_arr, null_offsets = value_data
                self.assertTrue(np.all(num_samples_arr == samples_per_message))
                values = np.concatenate([v[:num_samples_arr[i]] for i, v in enumerate(values_2d)], axis=None)

            self.assertEqual(samples_per_message * num_messages, values.shape[0])

            if np.issubdtype(values.dtype, np.integer):
                raw_v_t = VALUE_TYPES['INT64']
                encoded_v_t = VALUE_TYPES['DELTA_INT64']
            else:
                raw_v_t = VALUE_TYPES['DOUBLE']
                encoded_v_t = VALUE_TYPES['DOUBLE']

            encoded_bytes, w_headers, byte_start_array = \
                bc.encode_blocks(
                    gap_arr, values, freq_nhz, int(nom_times[0]),
                    raw_time_type=TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS'],
                    raw_value_type=raw_v_t,
                    encoded_time_type=TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS'],
                    encoded_value_type=encoded_v_t)

            # encode_blocks(times, values, freq_nhz: int, start_ns: int,
            # raw_time_type = None, raw_value_type = None, encoded_time_type = None,
            # encoded_value_type = None, scale_m: float = None, scale_b: float = None)

            r_time_data, r_value_data, r_headers = bc.decode_blocks(encoded_bytes, byte_start_array)

            if np.issubdtype(values.dtype, np.integer):
                self.assertTrue(np.array_equal(r_value_data, values))
            else:
                self.assertTrue(np.allclose(r_value_data, values, rtol=0.01, equal_nan=True))

            self.assertTrue(np.array_equal(gap_arr, r_time_data))


    def test_gap_arr_converter(self):
        num_messages = 10 ** 3

        for _ in range(100):
            value_data, nom_times, samples_per_message, freq_nhz = generate_random_sdk_compliant_data(num_messages)

            gap_arr = create_gap_arr(nom_times, samples_per_message, freq_nhz)

            if (10 ** 18) % freq_nhz == 0:
                self._test_time_recreation(nom_times, gap_arr, num_messages, samples_per_message, freq_nhz)

            interpreted_times = interpret_gap_arr(
                gap_arr, int(nom_times[0]), num_messages, samples_per_message, freq_nhz)

            self.assertTrue(np.array_equal(nom_times, interpreted_times))

    def _test_time_recreation(self, message_times, gap_arr, num_messages, samples_per_message, freq_nhz):
        self.assertEqual(0, (10 ** 18) % freq_nhz)
        sample_period = (10 ** 18) // freq_nhz

        stitched_time_array = np.concatenate(
            [np.arange(t, t + (sample_period * samples_per_message), sample_period)
             for t in message_times], axis=None)

        interpreted_time_array = interpret_gap_arr(
                gap_arr, int(message_times[0]), num_messages * samples_per_message, 1, freq_nhz)

        self.assertTrue(np.array_equal(stitched_time_array, interpreted_time_array))


if __name__ == '__main__':
    unittest.main()
