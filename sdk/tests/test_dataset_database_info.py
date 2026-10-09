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
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'database_info'
NO_TSC_DB_NAME = 'database_info_no_tsc'


def test_database_info():
    _test_for_both(DB_NAME, _test_database_info)


def _test_database_info(db_type, dataset_location, connection_params):
    AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    database_file = Path(dataset_location) / 'meta' / 'database.json'
    info = json.loads(database_file.read_text())
    if db_type == 'sqlite':
        assert info == {'type': 'sqlite', 'database': 'index.db'}
    else:
        assert info == {'type': db_type, 'host': connection_params['host'], 'port': connection_params['port'],
                        'database': connection_params['database']}

    # Opening a dataset is silent and leaves database.json as found, including for a dataset created without one.
    assert _open_silently(dataset_location, db_type, connection_params) == []

    database_file.unlink()
    assert _open_silently(dataset_location, db_type, connection_params) == []
    assert not database_file.exists()


def _open_silently(dataset_location, db_type, connection_params):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                  connection_params=connection_params)
    # Warnings raised by the SDK, or attributed to the code that opened it.
    return [w for w in caught if w.filename.endswith(('atrium_sdk.py', Path(__file__).name))]


def test_open_without_tsc_location(tmp_path, monkeypatch):
    _test_for_both(NO_TSC_DB_NAME, _test_open_without_tsc_location, tmp_path, monkeypatch)


def _test_open_without_tsc_location(db_type, dataset_location, connection_params, tmp_path, monkeypatch):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    device_id = sdk.insert_device(device_tag='device')
    other_device_id = sdk.insert_device(device_tag='other_device')
    measure_id = sdk.insert_measure(measure_tag='measure', freq=1, freq_units='Hz')

    freq_nhz = 10 ** 9
    times = np.arange(10, dtype=np.int64) * freq_nhz
    values = np.arange(10, dtype=np.float64)
    sdk.write_data_easy(measure_id, device_id, times, values, freq_nhz)

    # SQLite keeps its metadata inside the dataset, so it always needs the location.
    if db_type == 'sqlite':
        with pytest.raises(ValueError):
            AtriumSDK(metadata_connection_type=db_type, connection_params=connection_params)
        return

    monkeypatch.chdir(tmp_path)
    metadata_sdk = AtriumSDK(metadata_connection_type=db_type, connection_params=connection_params)

    # The metadata is fully available.
    assert measure_id in metadata_sdk.get_all_measures()
    assert device_id in metadata_sdk.get_all_devices()

    # Touching TSC files raises a clear error, and the working directory and database stay as they were. A write next
    # to an existing block reads that block first; a write for a new device creates a file.
    with pytest.raises(ValueError, match="TSC file location"):
        metadata_sdk.get_data(measure_id=measure_id, start_time_n=0, end_time_n=10 * freq_nhz, device_id=device_id)
    with pytest.raises(ValueError, match="TSC file location"):
        metadata_sdk.write_data_easy(measure_id, device_id, times + 100 * freq_nhz, values, freq_nhz)
    with pytest.raises(ValueError, match="TSC file location"):
        metadata_sdk.write_data_easy(measure_id, other_device_id, times, values, freq_nhz)
    assert list(tmp_path.iterdir()) == []
    assert len(metadata_sdk.get_interval_array(measure_id=measure_id, device_id=other_device_id)) == 0
