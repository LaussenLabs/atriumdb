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
        assert sdk.settings_dict['overwrite'] == 'ignore'

    finally:
        shutil.rmtree(TSC_DATASET_DIR)
