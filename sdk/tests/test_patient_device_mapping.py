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
from tests.testing_framework import _test_for_both

DB_NAME = 'test_device_patient_mapping'


def test_device_patient_mapping():
    _test_for_both(DB_NAME, _test_device_patient_mapping)


def _test_device_patient_mapping(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Pre-test setup: create patients, devices, and device-patient mappings
    patient_id_case1 = sdk.insert_patient(patient_id=123, mrn='123456')
    device_id_case1 = sdk.insert_device(device_tag='monitor001')

    start_time_case1 = 1647084000_000_000_000
    end_time_case1 = 1647094800_000_000_000
    sdk.insert_device_patient_data([(device_id_case1, patient_id_case1, start_time_case1, end_time_case1)])

    # Test different scenarios
    # 1. Test for valid mapping (should return the correct device ID)
    test_start_time = start_time_case1
    test_end_time = end_time_case1
    actual_device_id = sdk.convert_patient_to_device_id(test_start_time, test_end_time, patient_id_case1)
    assert actual_device_id == device_id_case1, f"Expected device_id: {device_id_case1}, Got: {actual_device_id}"

    # 2. Test for an out-of-range time span (should return None)
