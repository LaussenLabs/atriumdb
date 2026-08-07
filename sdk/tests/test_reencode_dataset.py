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
import pytest
import random

from atriumdb.adb_functions import convert_gap_data_to_timestamps, create_timestamps_from_gap_data, reencode_dataset
from tests.generate_wfdb import get_records
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset, TRUNCATED_SAMPLES_PER_RECORD
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-reencode'

MAX_RECORDS = 4
SEED = 42

# re-encoding is a format operation, indifferent to sample count. Truncate
# the waveform AND scale values_per_block by the same factor so the re-encoded
# data still spans the same number of blocks per measure (650,000/131,072 = 4.96;
# 20,000/4,096 = 4.88). Shrinking one without the other would collapse each
# measure to a single block and lose the multi-block re-encode path.
VALUES_PER_BLOCK = 4_096


# real parametrization instead of the _test_for_both helper -- both backends
# still run, but each gets its own test id ([sqlite] / [mariadb]), can be selected
# with -k/-m, reports its failure independently and shows up in --durations.
@pytest.mark.mitbih
@pytest.mark.parametrize("backend", parametrized_backends())
def test_reencode_dataset(backend):
    _test_reencode_dataset(*prepare_backend(DB_NAME, backend))


@pytest.mark.mitbih
@pytest.mark.parametrize("backend", parametrized_backends())
def test_reencode_dataset_period(backend):
    _test_reencode_dataset_period(*prepare_backend(DB_NAME, backend))


def _test_reencode_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED,
                             max_samples_per_record=TRUNCATED_SAMPLES_PER_RECORD)
    reencode_dataset(sdk, values_per_block=VALUES_PER_BLOCK, blocks_per_file=2048,
                     interval_gap_tolerance_nano=0)
    assert_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED,
                              max_samples_per_record=TRUNCATED_SAMPLES_PER_RECORD)


def _test_reencode_dataset_period(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED, use_period=True,
                             max_samples_per_record=TRUNCATED_SAMPLES_PER_RECORD)
    reencode_dataset(sdk, values_per_block=VALUES_PER_BLOCK, blocks_per_file=2048,
                     interval_gap_tolerance_nano=0)
    assert_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED, use_period=True,
                              max_samples_per_record=TRUNCATED_SAMPLES_PER_RECORD)
