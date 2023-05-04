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

from atriumdb import AtriumSDK
from adb_credentials import DLL_PATH, phillips_database_uri, PHILLIPS_TSC_FILE_LOCATION

from pathlib import Path
import shutil
import numpy as np
import time
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

NANO_IN_SECOND = 1_000_000_000

sdk = None


def process(measure, start, end, device):
    with sdk.sql_connect() as connection:
        headers, times, values = sdk.get_data(measure, start, end, device_id=device, analog=False,
                                              connection=connection)

        return values.size

    # headers, times, values = sdk.get_data(measure, start, end, device_id=device, analog=False)
    #
    # return values.size


def initializer(demo_name, top_level_dir):
    global sdk
    sdk = create_local_sdk_object(demo_name, top_level_dir, db_uri=None)
    sdk.mp_init()


def test_simple():
    top_level_dir = "/mnt/datasets"

    # Name the dataset
    demo_name = "simple_test_demo"

    # reset_demo(demo_name, top_level_dir)

    sdk = create_local_sdk_object(demo_name, top_level_dir, db_uri=None)
    original_dataset_info = write_simple_dataset(sdk, 'simple_test.pkl')
    # print("hello")
    #
    # headers, read_times, read_values = sdk.get_data(
    #     original_dataset_info['original_measure_id'],
    #     original_dataset_info['start_time'],
    #     original_dataset_info['end_time'],
    #     device_id=original_dataset_info['original_device_id'],
    #     analog=False)
    #
    # assert np.array_equal(read_times, original_dataset_info['original_time_arr'])
    # assert np.array_equal(read_values, original_dataset_info['original_scaled_value_data'])
    # _, expected_times, _ = sdk.get_data(
    #     original_dataset_info['original_measure_id'],
    #     original_dataset_info['start_time'],
    #     original_dataset_info['end_time'],
    #     device_id=original_dataset_info['original_device_id'],
    #     analog=False)

    bench_results = []
    read_size = None
    for _ in range(1):
        bench_start = time.perf_counter()
        # headers, read_times, read_values = sdk.get_data(original_dataset_info['original_measure_id'],
        #                                                 original_dataset_info['start_time'],
        #                                                 original_dataset_info['end_time'],
        #                                                 device_id=original_dataset_info['original_device_id'],
        #                                                 analog=False)

        for headers, read_times, read_values in \
                sdk.get_batched_data_generator(original_dataset_info['original_measure_id'],
                                               original_dataset_info['start_time'], original_dataset_info['end_time'],
                                               device_id=original_dataset_info['original_device_id'], analog=False):
            print(read_times, read_values)
        return

        bench_end = time.perf_counter()
        bench_results.append(bench_end - bench_start)
        read_size = read_values.size

    av_time = sum(bench_results) / len(bench_results)
    print(f"{read_size} in {av_time} seconds.")
    print(f"get data speed: {round(read_size / av_time, 2)} values per second.")
    return
    num_workers = 10
    measure_id = original_dataset_info['original_measure_id']
    device_id = original_dataset_info['original_device_id']
    start_time = original_dataset_info['start_time']
    end_time = original_dataset_info['end_time']

    total_duration = end_time - start_time
    worker_dur = (total_duration // num_workers) + 1

    starts = []
    ends = []
    worker_start = start_time
    for _ in range(num_workers):
        starts.append(worker_start)
        worker_start += worker_dur
        ends.append(worker_start)

    measures = [measure_id for _ in range(num_workers)]
    devices = [device_id for _ in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers, initializer=initializer, initargs=(demo_name, top_level_dir)) \
            as executor:
        start_bench = time.perf_counter()
        results = executor.map(process, measures, starts, ends, devices)
        num_values = sum(results)
        end_bench = time.perf_counter()

        print(f"{num_values} values in {round(end_bench - start_bench, 2)} seconds")
        print(f"{round(num_values / (end_bench - start_bench), 2)} values per second.")

    # assert np.array_equal(read_times, original_dataset_info['original_time_arr'])
    # assert np.allclose(read_values, original_dataset_info['original_scaled_value_data'])

    # reset_demo(demo_name, top_level_dir)


def create_local_sdk_object(demo_name, top_level_dir, db_uri=None):
    # Define where on disk the dataset will sit.
    highest_level_dir = f"{top_level_dir}/{demo_name}"
    db_path = f"{highest_level_dir}/meta/sdk_index.db"
    tsc_path = f"{highest_level_dir}/tsc"

    # Create the needed filepaths if they don't exist.
    Path(tsc_path).mkdir(parents=True, exist_ok=True)
    Path(f"{highest_level_dir}/meta").mkdir(parents=True, exist_ok=True)

    # Create an sqlite uri.
    if db_uri is None:
        db_uri = f"sqlite:///{db_path}"
    return AtriumSDK(dataset_location=tsc_path, atriumdb_lib_path=DLL_PATH)


def get_db_path(demo_name, top_level_dir):
    highest_level_dir = f"{top_level_dir}/{demo_name}"
    db_path = f"{highest_level_dir}/meta/sdk_index.db"
    return db_path


def get_tsc_path(demo_name, top_level_dir):
    highest_level_dir = f"{top_level_dir}/{demo_name}"
    tsc_path = f"{highest_level_dir}/tsc"
    return tsc_path


def write_simple_dataset(sdk, filename):
    if Path(filename).is_file():
        return pickle.load(open(filename, 'rb'))
    # Create some identifiers.
    example_measure_id = 1
    example_device_id = 1

    # Define Sample Rate.
    period_nano = int(0.002 * NANO_IN_SECOND)  # 2 ms
    freq_nano = (10 ** 18) // period_nano
    assert (10 ** 18) % period_nano == 0

    # add new identifiers
    sdk.insert_measure(measure_tag='sin_wave', freq_nhz=freq_nano, units=None)
    sdk.insert_device(device_tag='device_1')

    # Create a start time, and number of samples.
    start_time_nano = int(time.time() * NANO_IN_SECOND)
    num_samples = 13_000_000_000
    batch_size = 500_000_000

    # Create some gaps
    # [[index, duration], ...]
    gap_arr = [[10_095, 105 * NANO_IN_SECOND], [32_068, int(202.6 * NANO_IN_SECOND)],
               [95_834, int(96.932 * NANO_IN_SECOND)]]

    # Set some scale factors
    scale_m = 0.001
    scale_b = 0

    # Write Data
    batch_start_time = start_time_nano
    end_time_nano = -1
    for write_index in range(0, num_samples, batch_size):
        original_time_arr = np.arange(min(batch_size, num_samples - write_index))
        original_time_arr *= period_nano
        original_time_arr += batch_start_time

        for gap_index, gap_duration in gap_arr:
            original_time_arr[gap_index:] += gap_duration

        end_time_nano = original_time_arr[-1] + period_nano
        batch_start_time = end_time_nano

        # Create a sin wave value array
        original_value_data = np.sin(np.linspace(0, 8 * math.pi, num=original_time_arr.size))

        # Scale the data into quantized integers
        original_scaled_value_data = (original_value_data / scale_m).astype(np.int64)

        sdk.write_data_easy(example_measure_id, example_device_id,
                            original_time_arr,
                            original_scaled_value_data,
                            freq_nano, scale_m=scale_m, scale_b=scale_b)

    return_dict = \
        {
            # 'original_time_arr': original_time_arr,
            # 'original_value_data': original_value_data,
            # 'original_scaled_value_data': original_scaled_value_data,
            'start_time': start_time_nano,
            'end_time': end_time_nano,
            'period_nano': period_nano,
            'freq_nano': freq_nano,
            'scale_m': scale_m,
            'scale_b': scale_b,
            'original_device_id': example_device_id,
            'original_measure_id': example_measure_id}

    pickle.dump(return_dict, open(filename, 'wb'))

    return return_dict


def reset_demo(demo_name, top_level_dir):
    db_path = get_db_path(demo_name, top_level_dir)
    tsc_path = get_tsc_path(demo_name, top_level_dir)
    remove_all_files_in_directory(Path(tsc_path))
    delete_file(Path(db_path))


def remove_all_files_in_directory(dir_path):
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)


def delete_file(file_path: Path):
    file_path.unlink(missing_ok=True)


def reset_database(highest_level_dir):
    db_path = f"{highest_level_dir}/meta/sdk_index.db"
    tsc_path = f"{highest_level_dir}/tsc"

    Path(db_path).unlink(missing_ok=True)
    if Path(tsc_path).is_dir():
        shutil.rmtree(tsc_path)


if __name__ == "__main__":
    test_simple()
