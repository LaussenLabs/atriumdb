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
import pytest
import numpy as np
from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'nan_filled'


def test_nan_filled():
    _test_for_both(DB_NAME, _test_nan_filled)


def _test_nan_filled(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq_hz = 1

    measure_id = sdk.insert_measure("example", freq_hz, "units", freq_units="Hz")
    device_id = sdk.insert_device("example")

    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    times = np.array([1, 3, 4, 5, 8, 9], dtype=np.int64)
    scale_m, scale_b = 1.0, 0.0

    expected_nan_filled = np.array([1, np.nan, 2, 3, 4, np.nan, np.nan, 5, 6], dtype=np.float64)

    sdk.write_time_value_pairs(measure_id, device_id, times, data, freq=freq_hz, scale_m=scale_m, scale_b=scale_b,
                               freq_units="Hz", time_units="s")

    _, actual_nan_filled = sdk.get_data(measure_id, 1, 10, device_id=device_id, time_units="s", return_nan_filled=True)

    assert np.allclose(expected_nan_filled, actual_nan_filled, equal_nan=True)