from atriumdb.atrium_sdk import AtriumSDK
from pathlib import Path
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

TSC_DATASET_DIR = Path(__file__).parent / 'test_tsc_data' / 'settings_test'


def test_settings():
    try:
        TSC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        sdk = AtriumSDK.create_dataset(dataset_location=str(TSC_DATASET_DIR))
        assert sdk.settings_dict['protected_mode'] == 'True'
        assert sdk.settings_dict['overwrite'] == 'error'

    finally:
        shutil.rmtree(TSC_DATASET_DIR)
