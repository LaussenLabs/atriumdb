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
    patient_id_list = insert_random_patients(sdk_1, 100)

    institution_id = sdk_1.sql_handler.insert_institution(name="Test Institution")
    unit_id = sdk_1.sql_handler.insert_unit(institution_id=institution_id, name="Test Unit", unit_type="Test Type")
    bed_id = sdk_1.sql_handler.insert_bed(unit_id=unit_id, name="Test Bed")
    sdk_1.insert_encounter(start_time=1_000, end_time=2_000, patient_id=patient_id_list[0], bed_id=bed_id,
                           visit_number="V1", last_updated=3_000)
    sdk_1.insert_encounter(start_time=4_000, patient_id=patient_id_list[1], bed_id=bed_id, last_updated=5_000)

    # A bed with the same name in another unit
    other_unit_id = sdk_1.sql_handler.insert_unit(institution_id=institution_id, name="Other Unit",
                                                  unit_type="Test Type")
    other_bed_id = sdk_1.sql_handler.insert_bed(unit_id=other_unit_id, name="Test Bed")
    sdk_1.insert_encounter(start_time=6_000, end_time=7_000, patient_id=patient_id_list[2], bed_id=other_bed_id,
                           last_updated=8_000)

    # Test transfer_patients without deidentification
    # from_to_patient_id_dict = transfer_patients(sdk_1, sdk_2)
    device_id_map = transfer_devices(sdk_1, sdk_2)
    from_to_patient_id_dict = transfer_patient_info(sdk_1, sdk_2, patient_id_list="all", deidentify=False)


    all_patients_1 = sdk_1.get_all_patients()
    all_patients_2 = sdk_2.get_all_patients()

    for from_patient_id, to_patient_id in from_to_patient_id_dict.items():
        assert from_patient_id == to_patient_id
        assert all_patients_1[from_patient_id] == all_patients_2[to_patient_id]

    assert [e[1:] for e in sdk_2.get_encounters()] == [e[1:] for e in sdk_1.get_encounters()]

    # Test transfer_patients with deidentification
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)
    device_id_map = transfer_devices(sdk_1, sdk_2)
    time_shift = 10_000
    from_to_patient_id_dict_deid = transfer_patient_info(sdk_1, sdk_2, patient_id_list="all", deidentify=True,
                                                         time_shift_nano=time_shift)

    assert [e[1:] for e in sdk_2.get_encounters()] == [
        (from_to_patient_id_dict_deid[p_id], b_id, start + time_shift, None if end is None else end + time_shift,
         1, None, updated + time_shift) for _, p_id, b_id, start, end, _, _, updated in sdk_1.get_encounters()]

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

    # Only encounters overlapping a patient's time ranges are transferred
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)
    transfer_patient_info(sdk_1, sdk_2, patient_id_list="all", deidentify=False,
                          patient_time_ranges={patient_id_list[0]: [(1_500, 1_600)], patient_id_list[1]: [(0, 1_000)]})
    assert [e[1:] for e in sdk_2.get_encounters()] == \
        [e[1:] for e in sdk_1.get_encounters(patient_id=patient_id_list[0])]


def insert_random_patients(sdk, n):
    def random_string(size):
        return ''.join(random.choice(string.ascii_letters) for _ in range(size))

    def random_mrn():
        return str(random.randint(100000, 999999))

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
