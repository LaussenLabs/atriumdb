from atriumdb import AtriumSDK
import numpy as np
from pathlib import Path
import shutil


def test_dll():
    dataset_location = "./example_dir"
    reset_database(dataset_location)
    block_dll_path = Path(__file__).parent.parent / "bin" / "libTSC.dll"
    sdk = AtriumSDK(dataset_location=dataset_location, atriumdb_lib_path=block_dll_path)

    sdk.insert_device(None)
    sdk.insert_measure(None, 500)

    times = np.arange(1000, dtype="int64") * (10 ** 9)
    values = np.sin(times)

    # sdk.write_data_easy(1, 1, times, values, 10 ** 9)

    _, r_times, r_values = sdk.get_data(1, 0, 1000, device_id=1)

    print(r_times)
    print(r_values)


def reset_database(highest_level_dir):
    db_path = f"{highest_level_dir}/meta/index.db"
    tsc_path = f"{highest_level_dir}/tsc"

    Path(db_path).unlink(missing_ok=True)
    if Path(tsc_path).is_dir():
        shutil.rmtree(tsc_path)

