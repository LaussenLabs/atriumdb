from atriumdb.atrium_sdk import AtriumSDK
from pathlib import Path
import shutil

TSC_DATASET_DIR = Path(__file__).parent / 'test_tsc_data' / 'measure_device_inserts'


def test_measure_device_inserts():
    try:
        test_measure_dict = {
            1: ("abc", 500),
            2: ("def", 333),
            3: ("43", 1),
        }

        test_device_dict = {
            1: "dev_abc",
            2: "dev_def",
            3: "dev_3",
        }

        TSC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        sdk = AtriumSDK(dataset_location=str(TSC_DATASET_DIR))

    finally:
        shutil.rmtree(TSC_DATASET_DIR)
