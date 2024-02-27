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
from pathlib import Path

from atriumdb import AtriumSDK, DatasetDefinition
import shutil

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-export'
MAX_RECORDS = 1
SEED = 42


def test_export():
    _test_for_both(DB_NAME, _test_export)


def _test_export(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    device_patient_dict = write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measure_id_list = None
    device_id_list = None
    patient_id_list = None
    start = None
    end = None
    time_units = None
    batch_size = None

    measures = [measure_info['tag'] for measure_info in sdk_1.get_all_measures().values()]
    device_ids = {device_id: "all" for device_id in sdk_1.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    for export_format in ['tsc', 'csv', 'npz', 'wfdb']:
        original_path = Path(sdk_1.dataset_location)
        export_dataset_location = original_path.with_name(original_path.stem + f"_{export_format}")
        sdk_2 = AtriumSDK.create_dataset(dataset_location=export_dataset_location)
        transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False,
                      patient_info_to_transfer=None, include_labels=False, export_format=export_format)

    # assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, max_records=MAX_RECORDS, seed=SEED)
