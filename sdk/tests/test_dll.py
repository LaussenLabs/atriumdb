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

