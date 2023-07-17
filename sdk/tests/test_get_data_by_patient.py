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

from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = "atrium-get-data-patient"


def test_insert_data_with_patient_mapping():
    _test_for_both(DB_NAME, _test_insert_data_with_patient_mapping)


def _test_insert_data_with_patient_mapping(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_tag = 'signal_1'
    freq_hz = 1
    period = 1 / freq_hz

    freq_nhz = int(freq_hz * (10 ** 9))
    period_ns = int(period * (10 ** 9))

    device_tag = 'dev_1'

    measure_id = sdk.insert_measure(measure_tag, freq_hz, freq_units="Hz")
    device_id = sdk.insert_device(device_tag)

    num_values = 1_000_000
    og_time_data = np.arange(num_values, dtype=np.int64) * period_ns
    og_value_data = np.sin(np.arange(num_values))

    sdk.write_data_easy(measure_id, device_id, og_time_data, og_value_data, freq_hz)

    patient_id = insert_random_patients(sdk, 1)[0]

    # Map the device to the patient
    start_time = int(og_time_data[0])
    end_time = int(og_time_data[-1] + period_ns)
    sdk.insert_device_patient_data([(device_id, patient_id, start_time, end_time)])

    _, patient_read_times, patient_read_values = sdk.get_data(measure_id, start_time, end_time, patient_id=patient_id)

    assert np.array_equal(og_time_data, patient_read_times)
    assert np.array_equal(og_value_data, patient_read_values)
