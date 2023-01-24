from atriumdb import AtriumSDK
from pathlib import Path
import numpy as np
import time
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from adb_credentials import phillips_database_uri_client, natus_db_2_uri, DLL_PATH, PHILLIPS_TSC_FILE_LOCATION, \
    NATUS_FILE_LOC_2, phillips_database_uri, phillips_database_uri_async

dest_db_path = "/mnt/datasets/atriumdb_popular_local/meta/sdk_index.db"
dest_db_uri = f'sqlite:///{dest_db_path}'
dest_tsc_path = "/mnt/datasets/atriumdb_popular_local/tsc"

# sdk = AtriumSDK(database_uri=phillips_database_uri, block_codec_dll_path=DLL_PATH,
#                 tsc_file_location=PHILLIPS_TSC_FILE_LOCATION, session_string=None)
sdk = AtriumSDK(dataset_location=dest_tsc_path, database_uri=dest_db_uri)


def process(measure, start, end, device):
    with sdk.sql_connect() as connection:
        headers, times, values = sdk.get_data(measure, start, end, device_id=device, analog=False,
                                              connection=connection)

    return values.size


def initializer():
    global sdk
    sdk = AtriumSDK(dataset_location=dest_tsc_path, database_uri=dest_db_uri, atriumdb_lib_path=DLL_PATH)
    # sdk = AtriumSDK(database_uri=phillips_database_uri, block_codec_dll_path=DLL_PATH,
    #                 tsc_file_location=PHILLIPS_TSC_FILE_LOCATION, session_string=None)
    sdk.mp_init()


def test_mysql_mp():
    num_workers = 1

    # Where in time is the data located?
    measure_id = 3
    device_id = 74

    # gap_tolerance_nano means how large a gap in data there can exist and they data still be considered "contiguous".
    single_interval_arr = sdk.get_interval_array(measure_id,
                                                 device_id,
                                                 gap_tolerance_nano=5_000_000_000,
                                                 start=None, end=None)

    # Grab a month of ECG from device_id 74
    measure_id = 3
    device_id = 74

    start_time, end_time = None, None
    total_duration = 0
    one_month_in_nanoseconds = 30 * 24 * 60 * 60 * 1_000_000_000

    assert single_interval_arr.size > 0

    for start_interval, end_interval in single_interval_arr:
        if start_time is None:
            start_time = start_interval

        total_duration += end_interval - start_interval
        end_time = end_interval

        if total_duration >= 1 * one_month_in_nanoseconds:
            break

    # print(start_time, end_time, total_duration)
    # print("start")
    # # Pull one month of data
    # bench_start = time.perf_counter()
    # _, times, values = sdk.get_data(measure_id, start_time, end_time, device_id=device_id, analog=False)
    # bench_end = time.perf_counter()

    # print(values.size, bench_end-bench_start, values.size / (bench_end-bench_start))

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

    with ProcessPoolExecutor(max_workers=num_workers, initializer=initializer) \
            as executor:

        start_bench = time.perf_counter()
        results = executor.map(process, measures, starts, ends, devices)
        num_values = sum(results)
        end_bench = time.perf_counter()

        print(f"{num_values} values in {round(end_bench - start_bench, 2)} seconds")
        print(f"{round(num_values / (end_bench - start_bench), 2)} values per second.")
