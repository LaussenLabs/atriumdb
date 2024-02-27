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

import random
import shutil
import random
import time
import string

import names

import pytest

from atriumdb import AtriumSDK
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.devices import transfer_devices
from atriumdb.transfer.adb.patients import transfer_patient_info
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = "transfer-patients"


def test_transfer_patients():
    _test_for_both(DB_NAME, _test_transfer_patients)


def _test_transfer_patients(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Actual Test Below
    insert_random_patients(sdk_1, 100)

    # Test transfer_patients without deidentification
    # from_to_patient_id_dict = transfer_patients(sdk_1, sdk_2)
    device_id_map = transfer_devices(sdk_1, sdk_2)
    from_to_patient_id_dict = transfer_patient_info(sdk_1, sdk_2, patient_id_list="all", deidentify=False)


    all_patients_1 = sdk_1.get_all_patients()
    all_patients_2 = sdk_2.get_all_patients()

    for from_patient_id, to_patient_id in from_to_patient_id_dict.items():
        assert from_patient_id == to_patient_id
        assert all_patients_1[from_patient_id] == all_patients_2[to_patient_id]

    # Test transfer_patients with deidentification
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)
    device_id_map = transfer_devices(sdk_1, sdk_2)
    from_to_patient_id_dict_deid = transfer_patient_info(sdk_1, sdk_2, patient_id_list="all", deidentify=True)

    all_patients_1_deid = sdk_1.get_all_patients()
    all_patients_2_deid = sdk_2.get_all_patients()

    for from_patient_id, to_patient_id in from_to_patient_id_dict_deid.items():
        assert all_patients_1_deid[from_patient_id] != all_patients_2_deid[to_patient_id]
        assert all_patients_2_deid[to_patient_id] == {
            'id': to_patient_id,
            'mrn': None,
            'gender': None,
            'dob': None,
            'first_name': None,
            'middle_name': None,
            'last_name': None,
            'first_seen': None,
            'last_updated': None,
            'source_id': 1,
            'weight': None,
            'height': None
        }


def insert_random_patients(sdk, n):
    def random_string(size):
        return ''.join(random.choice(string.ascii_letters) for _ in range(size))

    def random_mrn():
        return random.randint(100000, 999999)

    def random_gender():
        return random.choice(['M', 'F', 'U'])

    def random_epoch():
        start = -2177452800_000_000_000
        end = 2524608000_000_000_000
        return random.randint(int(start), int(end))

    def random_height_weight():
        return random.uniform(0.5, 200.0)

    patient_id_list = []
    for _ in range(n):
        patient_id = sdk.insert_patient(mrn=random_mrn(),
                                        gender=random_gender(),
                                        dob=random_epoch(),
                                        first_name=names.get_first_name(),
                                        middle_name=random_string(1),
                                        last_name=names.get_last_name(),
                                        first_seen=random_epoch(),
                                        last_updated=random_epoch(),
                                        weight=random_height_weight(),
                                        height=random_height_weight(),
                                        weight_units='kg',
                                        height_units='cm')

        patient_id_list.append(patient_id)
        
    return patient_id_list
