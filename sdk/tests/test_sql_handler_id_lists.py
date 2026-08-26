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
"""
The ``... WHERE id IN (...)`` handler queries, on both backends.

These all route through ``SQLHandler._select_rows_in_list``. Two things they got wrong
before that helper existed, both of which these tests pin:

  * An EMPTY id list built the degenerate SQL ``IN ()``. SQLite quietly accepts that as
    "matches nothing"; MariaDB rejects it as a syntax error. So the same call returned
    ``[]`` on one backend and raised on the other. Asking for no ids is a legitimate
    query whose answer is "no rows", and both backends now say so.
  * ``select_blocks_by_ids`` compared ``len(rows)`` against the length of the raw id
    list rather than its DISTINCT length, so a repeated id looked like a missing row
    and raised ``Cannot find block_ids=set()`` -- an error naming nothing.

The absent-id error itself is deliberate and is pinned here too: both methods resolve
rows a block read is about to use, so a missing row must fail loudly at the lookup
rather than further downstream.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_sql_handler_id_lists.py -q
"""
import numpy as np
import pytest

from atriumdb import AtriumSDK

DB_NAME = 'atrium-id-lists'

FREQ_HZ = 1
PERIOD_NS = 1_000_000_000
NUM_VALUES = 64

# Every handler method that takes a list of ids, with the id list that selects nothing.
EMPTY_LIST_METHODS = [
    'select_all_measures_in_list',
    'select_all_devices_in_list',
    'select_all_beds_in_list',
    'select_all_units_in_list',
    'select_all_institutions_in_list',
    'select_all_sources_in_list',
    'select_all_device_encounters_by_encounter_list',
    'select_files',
    'select_blocks_by_ids',
]


def _sdk_with_one_block(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure('ecg', freq=FREQ_HZ, freq_units='Hz', units='mV')
    device_id = sdk.insert_device('dev_id_lists')

    times = np.arange(NUM_VALUES, dtype=np.int64) * PERIOD_NS
    values = np.arange(NUM_VALUES, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values, freq=FREQ_HZ, freq_units='Hz')

    return sdk, measure_id, device_id


@pytest.mark.parametrize('method_name', EMPTY_LIST_METHODS)
def test_empty_id_list_returns_no_rows(dataset_for_backend, method_name):
    """An empty id list is an empty result on BOTH backends, never a SQL syntax error."""
    sdk, _measure_id, _device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))

    assert getattr(sdk.sql_handler, method_name)([]) == []


def test_empty_keyword_filters_return_no_rows(dataset_for_backend):
    """The same rule for the methods whose id lists arrive as keyword filters."""
    sdk, _measure_id, device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))
    handler = sdk.sql_handler

    assert handler.select_all_patients_in_list(patient_id_list=[]) == []
    assert handler.select_all_patients_in_list(mrn_list=[]) == []
    assert handler.select_encounters(patient_id_list=[]) == []
    assert handler.select_encounters(mrn_list=[]) == []
    assert handler.select_blocks_for_device(device_id, []) == []
    assert handler.select_blocks_for_devices([], []) == []
    assert handler.select_blocks_for_devices([device_id], []) == []
    assert handler.select_blocks_for_devices([], [1]) == []


def test_absent_filter_still_means_no_filter(dataset_for_backend):
    """``None`` must keep meaning "everything" -- only an EMPTY list means "nothing".

    The empty-list guards sit next to these branches, so this is the regression that
    would catch one of them being written as a falsy check.
    """
    sdk, measure_id, device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))
    handler = sdk.sql_handler

    sdk.insert_patient(patient_id=1, mrn='mrn-1')

    # No filter at all -> every patient.
    assert len(handler.select_all_patients_in_list()) == 1
    # No measure filter -> every block for the device.
    assert len(handler.select_blocks_for_device(device_id)) == len(
        handler.select_blocks_for_device(device_id, [measure_id]))


def test_select_files_finds_the_rows_it_is_given(dataset_for_backend):
    sdk, measure_id, device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))

    block_list = sdk.sql_handler.select_blocks(int(measure_id), None, None, device_id, None)
    file_ids = sorted({row[3] for row in block_list})
    assert file_ids

    rows = sdk.sql_handler.select_files(file_ids)
    assert sorted(row[0] for row in rows) == file_ids


def test_select_files_raises_on_an_absent_id(dataset_for_backend):
    sdk, _measure_id, _device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))

    with pytest.raises(RuntimeError, match=r"Cannot find file_ids=.*9999"):
        sdk.sql_handler.select_files([9999])


def test_select_blocks_by_ids_tolerates_a_repeated_id(dataset_for_backend):
    """A duplicate id is one row, not a missing one."""
    sdk, measure_id, device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))

    block_list = sdk.sql_handler.select_blocks(int(measure_id), None, None, device_id, None)
    block_id = block_list[0][0]

    rows = sdk.sql_handler.select_blocks_by_ids([block_id, block_id])

    assert len(rows) == 1
    assert rows[0][0] == block_id


def test_select_blocks_by_ids_raises_on_an_absent_id(dataset_for_backend):
    sdk, _measure_id, _device_id = _sdk_with_one_block(*dataset_for_backend(DB_NAME))

    with pytest.raises(RuntimeError, match=r"Cannot find block_ids=.*9999"):
        sdk.sql_handler.select_blocks_by_ids([9999])
