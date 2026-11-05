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
import pytest

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
    _test_for_both(DB_NAME, _test_transfer_period)
    _test_for_both(DB_NAME, _test_transfer_without_re_encoding)
    _test_for_both(DB_NAME, _test_transfer_labels)


def _test_transfer(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(
        measures=measures, device_ids=device_ids,
        labels=[label_name_info['name'] for label_name_info in sdk_1.get_all_label_names().values()])


    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, reencode_waveforms=True)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, max_records=MAX_RECORDS, seed=SEED)


def _test_transfer_labels(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    ecg_id = sdk_1.insert_measure("ECG", 500, "mV", freq_units="Hz")
    pleth_id = sdk_1.insert_measure("PLETH", 125, "%", freq_units="Hz")
    device_id = sdk_1.insert_device("monitor_1")
    sdk_1.write_data_easy(ecg_id, device_id, np.arange(5000) / 500, np.arange(5000), 500, time_units="s", freq_units="Hz")
    sdk_1.insert_labels([("arrhythmia", device_id, ecg_id, "cardiologist", 1, 2),
                         ("artifact", device_id, None, None, 3, 4)], time_units="s", source_type="device_id")

    definition = DatasetDefinition(measures=["ECG"], device_ids={device_id: "all"}, labels=["arrhythmia", "artifact"])
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, include_labels=True)

    dest_ecg_id = sdk_2.get_measure_id("ECG", 500, "mV", freq_units="Hz")
    labels = sorted((label['label_name'], label['measure_id'], label['label_source'], label['start_time_n'],
                     label['end_time_n']) for label in sdk_2.get_labels())
    assert labels == [("arrhythmia", dest_ecg_id, "cardiologist", 1_000_000_000, 2_000_000_000),
                      ("artifact", None, None, 3_000_000_000, 4_000_000_000)]

    # A label on a measure left out of the definition cannot be transferred
    sdk_1.insert_labels([("arrhythmia", device_id, pleth_id, None, 5, 6)], time_units="s", source_type="device_id")
    with pytest.raises(ValueError, match="not in the dataset definition"):
        transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, include_labels=True)


def _test_transfer_period(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED, use_period=True)

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(
        measures=measures, device_ids=device_ids,
        labels=[label_name_info['name'] for label_name_info in sdk_1.get_all_label_names().values()])


    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, reencode_waveforms=True)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, max_records=MAX_RECORDS, seed=SEED, use_period=True)

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
