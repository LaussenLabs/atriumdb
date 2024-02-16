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
import pytest

DB_NAME_INSERT_WITH_ID = 'test_db_insert_with_id'
DB_NAME_INSERT_WITHOUT_ID = 'test_db_insert_without_id'


def test_insert_patient_with_id():
    _test_for_both(DB_NAME_INSERT_WITH_ID, _test_insert_patient_with_id)


def test_insert_patient_without_id():
    _test_for_both(DB_NAME_INSERT_WITHOUT_ID, _test_insert_patient_without_id)


def _test_insert_patient_with_id(db_type, dataset_location, connection_params):
    assigned_patient_id = 123

    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk.insert_patient(patient_id=assigned_patient_id, mrn="123456", gender="M",
                       dob=946684800000000000, first_name="John", middle_name="A",
                       last_name="Doe", first_seen=1609459200000000000, last_updated=1609459200000000000, source_id=1,
                       weight=70, weight_units='kg', height=180, height_units='cm')

    patient_info = sdk.get_patient_info(patient_id=assigned_patient_id)

    assert patient_info['id'] == assigned_patient_id
    assert patient_info['mrn'] == 123456
    assert patient_info['gender'] == "M"
    assert patient_info['dob'] == 946684800000000000
    assert patient_info['first_name'] == "John"
    assert patient_info['middle_name'] == "A"
    assert patient_info['last_name'] == "Doe"
    assert patient_info['first_seen'] == 1609459200000000000
    assert patient_info['last_updated'] == 1609459200000000000
    assert patient_info['source_id'] == 1
    assert patient_info['weight'] == 70
    assert patient_info['height'] == 180


def _test_insert_patient_without_id(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    inserted_patient_id = sdk.insert_patient(mrn=654321, gender="F", dob=946684800000000000, first_name="Jane",
                                             middle_name="B", last_name="Smith", first_seen=1609459200000000000,
                                             last_updated=1609459200000000000, source_id=1,
                                             weight=55, weight_units='kg', height=165, height_units='cm')

    assert isinstance(inserted_patient_id, int) and inserted_patient_id > 0

    patient_info = sdk.get_patient_info(mrn=654321)

    assert patient_info['id'] == inserted_patient_id
    assert patient_info['mrn'] == 654321
    assert patient_info['gender'] == "F"
    assert patient_info['dob'] == 946684800000000000
    assert patient_info['first_name'] == "Jane"
    assert patient_info['middle_name'] == "B"
    assert patient_info['last_name'] == "Smith"
    assert patient_info['first_seen'] == 1609459200000000000
    assert patient_info['last_updated'] == 1609459200000000000
    assert patient_info['source_id'] == 1
    assert patient_info['weight'] == 55
    assert patient_info['height'] == 165
