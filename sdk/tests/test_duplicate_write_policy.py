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
"""Two write entry points disagree about which duplicate value is authoritative.

Write ``[(t, 1.0), (t, 2.0), (t+1s, 3.0)]`` into two identical datasets, one through
``write_time_value_pairs`` and one through ``write_data_easy``, then read both back with
``allow_duplicates=False``:

===========================  ==============
``write_time_value_pairs``   ``[1.0, 3.0]``
``write_data_easy``          ``[2.0, 3.0]``
===========================  ==============

The design ruling for this dataset is that **duplicates are a read-side concern**:
storage keeps what it was given, and ``allow_duplicates=False`` picks a survivor
according to the dataset's ``overwrite`` policy, which by default is that the newer
value wins. ``write_data_easy`` follows that. ``write_time_value_pairs`` collapses the
duplicate at *write* time via ``np.unique``, which keeps the first occurrence -- so the
losing value is not merely deselected, it is never stored, and no read-side policy can
recover it.

The reversal in ``_write_time_value_pairs_to_dataset`` that makes "push order decides
who wins" true is applied to whole pushed dictionaries, which is right for duplicates
*between* buffered pushes and does nothing for a duplicate *inside* one caller-supplied
array.

Marked ``xfail(strict=True)``: it fails today and is the specification of the fix. When
the two paths are reconciled this becomes an XPASS failure, forcing the marker to be
removed deliberately rather than left behind asserting nothing.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_duplicate_write_policy.py -q -rx
"""
import numpy as np
import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-duplicate-write-policy'

SEC = 10 ** 9
BASE = 1_700_000_000 * SEC

TIMES = np.array([BASE, BASE, BASE + SEC], dtype=np.int64)
VALUES = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Under the default 'ignore'/overwrite policy the newer value for a repeated timestamp
# wins, so the survivor at BASE is 2.0 -- the second of the two the caller supplied.
EXPECTED_SURVIVORS = [2.0, 3.0]


def _write_and_read(db_type, dataset_location, connection_params, writer):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    measure_id = sdk.insert_measure("hr", 1, freq_units="Hz", units="bpm")
    device_id = sdk.insert_device("dev_dupe")
    writer(sdk, measure_id, device_id)
    _, _times, values = sdk.get_data(
        measure_id, BASE - SEC, BASE + 5 * SEC, device_id=device_id, allow_duplicates=False)
    sdk.close()
    return values.tolist()


def _via_time_value_pairs(sdk, measure_id, device_id):
    sdk.write_time_value_pairs(measure_id, device_id, TIMES, VALUES)


def _via_write_data_easy(sdk, measure_id, device_id):
    sdk.write_data_easy(measure_id, device_id, TIMES, VALUES, freq=1, freq_units="Hz")


@pytest.mark.xfail(strict=True, reason=
                   "write_time_value_pairs collapses a duplicate timestamp inside one "
                   "caller-supplied array with np.unique, keeping the FIRST value, so the "
                   "newer value is never stored and the dataset's overwrite policy never "
                   "gets to choose")
@pytest.mark.parametrize("backend", parametrized_backends())
def test_write_paths_agree_on_the_duplicate_survivor(backend):
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME, backend)

    pairs = _write_and_read(db_type, f"{dataset_location}_pairs",
                            _params(connection_params, "pairs"), _via_time_value_pairs)
    easy = _write_and_read(db_type, f"{dataset_location}_easy",
                           _params(connection_params, "easy"), _via_write_data_easy)

    assert pairs == easy, (
        f"write_time_value_pairs stored {pairs} and write_data_easy stored {easy} "
        f"for identical input under an identical policy")
    assert pairs == EXPECTED_SURVIVORS, (
        f"expected the newer value to win under the default policy, got {pairs}")


def _params(connection_params, suffix):
    """A distinct MariaDB database per writer; SQLite needs nothing but the path."""
    if not connection_params:
        return connection_params
    params = dict(connection_params)
    params['database'] = f"{connection_params['database']}-{suffix}"
    return params
