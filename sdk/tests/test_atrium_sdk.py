import unittest
import random
import numpy as np
import os.path
import multiprocessing
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

atrium_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(atrium_path)

from generate_data import generate_random_sdk_compliant_data
from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.block import create_gap_arr, interpret_gap_arr
from atriumdb.helpers.block_constants import TIME_TYPES, VALUE_TYPES


def register_measure_device_ids(atrium_sdk, device_id, measure_ids):
    for measure_id in measure_ids:
        with atrium_sdk.sql_api.begin() as conn:
            atrium_sdk.sql_api.insert_measure_id(conn, 500 * (10 ** 9), id=measure_id, measure_tag=str(measure_id))
    with atrium_sdk.sql_api.begin() as conn:
        atrium_sdk.sql_api.insert_device_id(conn, id=device_id, device_tag=str(device_id))


def write_test_data_multiple_times(atrium_sdk, chunk_sizes, device_id, encoded_v_t, freq_nhz, lock,
                                   measure_ids, nom_times, num_messages, raw_v_t, samples_per_message, t_t, values):
    # Write it in three ways to three measure ids
    for measure_id, chunk_size in zip(measure_ids, chunk_sizes):
        print("write", measure_id, "spm", samples_per_message, "v_type", values.dtype, "chunk_size", chunk_size)

        for messages_written in tqdm(range(0, num_messages, chunk_size)):
            value_chunk = \
                values[messages_written * samples_per_message:
                       (messages_written + chunk_size) * samples_per_message]

            time_chunk = nom_times[messages_written:messages_written + chunk_size]

            gap_arr = create_gap_arr(time_chunk, samples_per_message, freq_nhz)
            atrium_sdk.write_data(measure_id, device_id, gap_arr, value_chunk, freq_nhz, int(time_chunk[0]),
                                  raw_time_type=t_t, raw_value_type=raw_v_t, encoded_time_type=t_t,
                                  encoded_value_type=encoded_v_t)

            messages_written += chunk_size


def decide_samples_per_messsage():
    if random.random() < 0.333:
        samples_per_message = 1
    else:
        samples_per_message = None
    return samples_per_message


class TestAtriumSDK(unittest.TestCase):
    def test_total_system(self):
        num_messages = 10 ** 4

        database_loc = str((Path(atrium_path) / "data" / "test.db").resolve())
        database_uri = 'sqlite:///{}'.format(database_loc)
        binary_path = str((Path(atrium_path) / "data").resolve())

        if Path(binary_path).exists():
            shutil.rmtree(binary_path)

        Path(binary_path).mkdir(parents=True, exist_ok=True)

        atrium_sdk = AtriumSDK(dataset_location=binary_path)

        measure_ids = [1, 2, 3]
        chunk_sizes = [atrium_sdk.block.block_size // 100,
                       atrium_sdk.block.block_size,
                       atrium_sdk.block.block_size * 100]

        device_id = 1337

        register_measure_device_ids(atrium_sdk, device_id, measure_ids)

        lock = None
        start_time = 0

        for loop_i in tqdm(range(50)):
            # Generate Data
            samples_per_message = decide_samples_per_messsage()
            encoded_v_t, freq_nhz, nom_times, raw_v_t, samples_per_message, t_t, values = self.prepare_test_data(
                num_messages, samples_per_message, start_time)

            write_test_data_multiple_times(atrium_sdk, chunk_sizes, device_id, encoded_v_t, freq_nhz,
                                           lock, measure_ids, nom_times, num_messages, raw_v_t,
                                           samples_per_message, t_t, values)

            is_integer_times = (10 ** 18) % freq_nhz == 0

            if is_integer_times:
                period_ns = (10 ** 18) // freq_nhz
                expected_times = \
                    np.concatenate([np.arange(t, t + (samples_per_message * period_ns), period_ns)
                                    for t in nom_times], axis=None)
            else:
                message_period_ns = (samples_per_message * (10 ** 18)) // freq_nhz
                expected_times = \
                    np.concatenate([np.linspace(t, t + message_period_ns,
                                                num=samples_per_message, endpoint=False)
                                    for t in nom_times], axis=None)

            # Read the entire portion
            for measure_id in measure_ids:
                # print("read", measure_id)
                r_times, r_values, headers = \
                    atrium_sdk.get_data(measure_id, nom_times[0],
                                        nom_times[-1] + (2 * samples_per_message * ((10 ** 18) // freq_nhz)), device_id,
                                        auto_convert_gap_to_time_array=True)

                self.assertTrue(np.allclose(r_times, expected_times))

                self.assertTrue(np.array_equal(r_values, values))

                for _ in range(50):
                    left = random.randint(0, expected_times.size - 2)
                    right = random.randint(left, expected_times.size - 1)
                    ex_left, ex_right = expected_times[left], expected_times[right]

                    print("ex_left, ex_right", ex_left, ex_right)
                    r_times, r_values, headers = \
                        atrium_sdk.get_data(measure_id, ex_left, ex_right, device_id,
                                            auto_convert_gap_to_time_array=True)
                    print("r_times[0], r_times[-1]", r_times[0], r_times[-1])
                    print()
                    self.assertTrue(np.allclose(r_times, expected_times[left:right+1]))

                    self.assertTrue(np.array_equal(r_values, values[left:right+1]))

                # Increase the start time
                start_time = nom_times[-1] + (2 * samples_per_message * ((10 ** 18) // freq_nhz))

        # Delete all tsc files and sqlite db.
        shutil.rmtree(binary_path)

    def prepare_test_data(self, num_messages, samples_per_message, start_time):
        value_data, nom_times, samples_per_message, freq_nhz = \
            generate_random_sdk_compliant_data(num_messages,
                                               samples_per_message=samples_per_message,
                                               start_time=start_time)
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
        t_t = TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']
        return encoded_v_t, freq_nhz, nom_times, raw_v_t, samples_per_message, t_t, values


if __name__ == '__main__':
    unittest.main()
