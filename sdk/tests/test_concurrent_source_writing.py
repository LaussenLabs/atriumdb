from atriumdb.atrium_sdk import AtriumSDK
from pathlib import Path
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

TSC_DATASET_DIR = Path(__file__).parent / 'test_tsc_data' / 'concurrent_measure_device_inserts'

process_sdk = None


def test_concurrent_source_writing():
    try:
        test_measure_list = [(str(measure_id), random.randint(1, 2048)) for measure_id in range(1000)]

        test_device_list = [str(device_id) for device_id in range(1000)]

        num_processes = 4

        TSC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        sdk = AtriumSDK(dataset_location=str(TSC_DATASET_DIR))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for _ in range(num_processes):
                futures.append(executor.submit(write_source_info_process, test_measure_list, test_device_list))

            for future in as_completed(futures):
                future.result()

        for (measure_tag, freq_hz) in test_measure_list:
            measure_id = sdk.get_measure_id(measure_tag, freq_hz, None, freq_units="Hz")
            assert measure_id is not None

        for device_tag in test_device_list:
            device_id = sdk.get_device_id(device_tag)
            assert device_id is not None

        read_measure_list = []
        for measure_id in sdk.get_all_measure_ids():
            measure_info_dict = sdk.sql_api.measure_id_dict[measure_id]
            read_measure_list.append((measure_info_dict['measure_tag'], sdk.get_freq(measure_id, freq_units="Hz")))

        read_device_list = []
        for device_id in sdk.get_all_device_ids():
            device_info_dict = sdk.sql_api.device_id_dict[device_id]
            read_device_list.append(device_info_dict['device_tag'])

        assert sorted(read_measure_list) == sorted(test_measure_list)
        assert sorted(read_device_list) == sorted(test_device_list)

    finally:
        shutil.rmtree(TSC_DATASET_DIR)
        pass


def write_source_info_process(measure_list, device_list):
    global process_sdk
    if process_sdk is None:
        process_sdk = AtriumSDK(dataset_location=str(TSC_DATASET_DIR))

    for (measure_tag, freq_hz) in measure_list:
        process_sdk.insert_measure(measure_tag, freq_hz, None, freq_units="Hz")

    for device_tag in device_list:
        process_sdk.insert_device(device_tag)
