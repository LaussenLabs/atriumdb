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


def test_get_data_by_patient_clipped_to_mapping():
    _test_for_both(f"{DB_NAME}-clipped", _test_get_data_by_patient_clipped_to_mapping)


def _test_get_data_by_patient_clipped_to_mapping(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure('signal_1', 1, freq_units="Hz")
    device_id = sdk.insert_device('dev_1')
    period_ns = 1_000_000_000
    times = np.arange(100, dtype=np.int64) * period_ns
    sdk.write_data_easy(measure_id, device_id, times, np.arange(100, dtype=np.float64), 1, freq_units="Hz")

    # The first patient has two overlapping mappings, then the device moves to the second patient at 50 s.
    first_patient_id, second_patient_id = sdk.insert_patient(), sdk.insert_patient()
    sdk.insert_device_patient_data([(device_id, first_patient_id, 0, 30 * period_ns),
                                    (device_id, first_patient_id, 20 * period_ns, 50 * period_ns),
                                    (device_id, second_patient_id, 50 * period_ns, 100 * period_ns)])

    _, read_times, read_values = sdk.get_data(measure_id, 0, 60 * period_ns, patient_id=first_patient_id)
    assert np.array_equal(read_times, times[:50])
    assert np.array_equal(read_values, np.arange(50))

    _, nan_filled_values = sdk.get_data(measure_id, 40 * period_ns, 60 * period_ns, patient_id=first_patient_id,
                                        return_nan_filled=True)
    assert np.array_equal(nan_filled_values[:10], np.arange(40, 50))
    assert np.all(np.isnan(nan_filled_values[10:]))

    # Off the sample grid, each slot holds its nearest sample, and the sample at 50 s goes to the second patient.
    for offset, first_values, second_values in [(period_ns // 3, np.arange(40, 50), np.arange(50, 60)),
                                                (2 * period_ns // 3, np.arange(41, 50), np.arange(50, 61))]:
        for patient_id, expected_values in [(first_patient_id, first_values), (second_patient_id, second_values)]:
            _, nan_filled_values = sdk.get_data(measure_id, 40 * period_ns + offset, 60 * period_ns + offset,
                                                patient_id=patient_id, return_nan_filled=True)
            assert np.array_equal(nan_filled_values[~np.isnan(nan_filled_values)], expected_values)


def test_get_data_by_device_and_patient():
    _test_for_both(f"{DB_NAME}-device-and-patient", _test_get_data_by_device_and_patient)


def _test_get_data_by_device_and_patient(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure('signal_1', 1, freq_units="Hz")
    device_1, device_2 = sdk.insert_device('dev_1'), sdk.insert_device('dev_2')
    period_ns = 1_000_000_000
    times = np.arange(100, dtype=np.int64) * period_ns
    sdk.write_data_easy(measure_id, device_1, times, np.arange(100, dtype=np.float64), 1, freq_units="Hz")
    sdk.write_data_easy(measure_id, device_2, times, np.arange(100, dtype=np.float64) + 1000, 1, freq_units="Hz")

    # The first patient moves from device 1 to device 2 at 50 s, when the second patient takes device 1.
    first_patient_id, second_patient_id = sdk.insert_patient(), sdk.insert_patient()
    sdk.insert_device_patient_data([(device_1, first_patient_id, 0, 50 * period_ns),
                                    (device_2, first_patient_id, 50 * period_ns, 100 * period_ns),
                                    (device_1, second_patient_id, 50 * period_ns, 100 * period_ns)])

    _, read_times, read_values = sdk.get_data(measure_id, 0, 100 * period_ns, device_id=device_1,
                                              patient_id=first_patient_id)
    assert np.array_equal(read_times, times[:50])
    assert np.array_equal(read_values, np.arange(50))
