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

from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, create_gap_arr, merge_gap_data
import numpy as np
import random

from atriumdb.adb_functions import convert_gap_data_to_timestamps, create_timestamps_from_gap_data, reencode_dataset
from tests.generate_wfdb import get_records
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-reencode'

MAX_RECORDS = 4
SEED = 42


def test_reencode_dataset():
    _test_for_both(DB_NAME, _test_reencode_dataset)


def _test_reencode_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)
    reencode_dataset(sdk, values_per_block=131072, blocks_per_file=2048, interval_gap_tolerance_nano=0)
    assert_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)
