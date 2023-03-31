from atriumdb import AtriumSDK
from adb_credentials import DLL_PATH, DUMMY_FILE_LOC, dummy_uri

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

from tests.test_simple import create_local_sdk_object, write_simple_dataset, remove_all_files_in_directory

logging_level = logging.INFO

logging.basicConfig(
    level=logging_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

NANO_IN_SECOND = 1_000_000_000

sdk = None


def test_mp():
    num_workers = 10
    num_slices = 10
    top_level_dir = "/mnt/datasets"

    # Name the dataset
    demo_name = "simple_test_demo"
    driver_str = "pymysql"

    # remove_all_files_in_directory(DUMMY_FILE_LOC)
    write_sdk = AtriumSDK(dataset_location=DUMMY_FILE_LOC)
    original_dataset_info = write_simple_dataset(write_sdk, "mp_test.pkl")
    del write_sdk
    total_duration = original_dataset_info['end_time'] - original_dataset_info['start_time']
    total_duration //= 1
    worker_dur = (total_duration // num_slices) + 1

    starts = []
    ends = []
    worker_start = original_dataset_info['start_time']
    for _ in range(num_slices):
        starts.append(worker_start)
        worker_start += worker_dur
        ends.append(worker_start)

    measures = [original_dataset_info['original_measure_id'] for _ in range(num_slices)]
    devices = [original_dataset_info['original_device_id'] for _ in range(num_slices)]

    with ProcessPoolExecutor(max_workers=num_workers, initializer=initializer) \
            as executor:
        start_bench = time.perf_counter()
        results = executor.map(process, measures, starts, ends, devices)
        num_values = sum(results)
        end_bench = time.perf_counter()

        print(f"{num_values} values in {round(end_bench - start_bench, 2)} seconds")
        print(f"{round(num_values / (end_bench - start_bench), 2)} values per second.")


def process(measure, start, end, device):
    with sdk.sql_connect() as connection:
        headers, times, values = sdk.get_data(
            measure,
            start,
            end,
            device_id=device,
            analog=False,
            connection=connection)

    return values.size


def initializer():
    global sdk
    sdk = AtriumSDK(dataset_location=DUMMY_FILE_LOC, atriumdb_lib_path=DLL_PATH)
    sdk.mp_init()
