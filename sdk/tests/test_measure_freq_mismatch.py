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
import warnings

import pytest
import numpy as np
from atriumdb import AtriumSDK, DatasetDefinition
from tests.testing_framework import _test_for_both

DB_NAME = 'measure_freq_mismatch_test'


def test_measure_freq_mismatch():
    _test_for_both(DB_NAME, _test_measure_freq_mismatch)


def _test_measure_freq_mismatch(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    device_id = sdk.insert_device("test_device")
    values = np.arange(1000)
    times = np.arange(values.size) / 125

    # freq_units omitted: the measure is stored at 125 nHz.
    nhz_measure_id = sdk.insert_measure("ecg_nhz", 125, "mV")
    with pytest.warns(UserWarning, match=r"125 Hz to measure 'ecg_nhz' \(id \d+\), which is stored at 125 nHz"):
        sdk.write_segment(nhz_measure_id, device_id, values, 0, freq=125, time_units="s")
    with pytest.warns(UserWarning, match="stored at 125 nHz"):
        sdk.write_time_value_pairs(nhz_measure_id, device_id, times + 100, values, time_units="s")

    definition = DatasetDefinition(measures=["ecg_nhz"], device_tags={"test_device": "all"})
    with pytest.raises(ValueError, match=r"measure 'ecg_nhz' \(id \d+\), stored at 125 nHz"):
        sdk.get_iterator(definition, 30, 30, time_units="s")

    hz_measure_id = sdk.insert_measure("ecg_hz", 125, "mV", freq_units="Hz")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdk.write_segment(hz_measure_id, device_id, values, 0, freq=125, time_units="s")
        sdk.write_time_value_pairs(hz_measure_id, device_id, times + 100, values, time_units="s")
    assert not any("stored at" in str(w.message) for w in caught)
