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
import numpy as np

from atriumdb import AtriumSDK, DatasetDefinition, partition_dataset
import shutil

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-transfer'
MAX_RECORDS = 10
SEED = 42


def test_transfer():
    _test_for_both(DB_NAME, _test_transfer)
    # _test_for_both(DB_NAME, _test_transfer_without_re_encoding)
    # _test_for_both(DB_NAME, _test_transfer_with_patient_context)
    # _test_for_both(DB_NAME, _test_transfer_with_patient_context_deidentify_timeshift)


def _test_transfer(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk_1.get_all_devices().keys()}
    label_name = list(sdk_1.get_all_label_names().values())[0]['name']
    definition = DatasetDefinition(measures=measures, device_ids=device_ids, labels=[label_name])

    # Test the dataset splitter
    train_def, test_def, val_def = partition_dataset(
        definition,
        sdk_1,
        partition_ratios=[60, 20, 20],
        priority_stratification_labels=[label_name],
        random_state=SEED,
        verbose=False
    )

    test_patients = list(test_def.data_dict['patient_ids'].keys())
    train_patients = list(train_def.data_dict['patient_ids'].keys())
    val_patients = list(val_def.data_dict['patient_ids'].keys())

    assert not (set(test_patients) & set(train_patients)), "Overlap found between test and train sets"
    assert not (set(test_patients) & set(val_patients)), "Overlap found between test and validation sets"
    assert not (set(train_patients) & set(val_patients)), "Overlap found between train and validation sets"

    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, max_records=MAX_RECORDS, seed=SEED)


def _test_transfer_without_re_encoding(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, reencode_waveforms=False)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, max_records=MAX_RECORDS, seed=SEED)


def _test_transfer_with_patient_context(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)


    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {device_id: "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False)

    assert_mit_bih_to_dataset(
        sdk_2, device_patient_map=device_patient_dict, use_patient_id=True, max_records=MAX_RECORDS, seed=SEED)


def _test_transfer_with_patient_context_deidentify_timeshift(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Test
    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {device_id: "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, time_shift=500, reencode_waveforms=True)

    assert_mit_bih_to_dataset(
        sdk_2, device_patient_map=device_patient_dict, deidentify=True, time_shift=-500, max_records=MAX_RECORDS,
        seed=SEED)
