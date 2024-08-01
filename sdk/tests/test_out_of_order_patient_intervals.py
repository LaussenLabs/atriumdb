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

DB_NAME = 'out_of_order_patient_intervals'


def test_out_of_order_patient_intervals():
    _test_for_both(DB_NAME, _test_out_of_order_patient_intervals)


def _test_out_of_order_patient_intervals(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq_hz = 1
    freq_nhz = (10 ** 9) * freq_hz
    period_ns = (10 ** 9) // freq_hz
    measure_id = sdk.insert_measure("sine", freq_nhz, "Unitless")
    patient_id_1 = sdk.insert_patient()

    device_id_1 = sdk.insert_device("dev1")
    device_id_2 = sdk.insert_device("dev2")

    ingest_metadata = [
        # Device, Patient, Start_Epoch_s, End_Epoch_s
        (device_id_1, patient_id_1, 5_000, 10_000),
        (device_id_2, patient_id_1, 1_000, 3_000),
    ]

    for device_id, patient_id, start_epoch_s, end_epoch_s in ingest_metadata:
        start_nano, end_nano = int(start_epoch_s * (10 ** 9)), int(end_epoch_s * (10 ** 9))

        # insert device_patient data
        sdk.insert_device_patient_data([(device_id, patient_id, start_nano, end_nano)])

        # Create wave data
        times = np.arange(start_nano, end_nano, period_ns, dtype=np.int64)
        values = 10_000 * np.sin(2 * np.pi * freq_hz * times)
        int_values = values.astype(np.int64)

        sdk.write_data_easy(measure_id, device_id, times, int_values, freq_nhz)

    expected_interval_array = np.array([
        [1_000 * 10**9, 3_000 * 10**9],
        [5_000 * 10 ** 9, 10_000 * 10 ** 9],
    ], dtype=np.int64)

    actual_interval_array = sdk.get_interval_array(measure_id, patient_id=patient_id_1)

    assert np.array_equal(actual_interval_array, expected_interval_array)

