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
import shutil
from pathlib import Path
import os

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler

from atriumdb.transfer.formats.dataset import export_dataset, import_dataset
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both


DB_NAME = 'atrium-formats-transfer'
PARTIAL_DB_NAME = 'atrium-formats-partial-transfer'
MAX_RECORDS = 1
SEED = 42


def test_csv_dataset():
    for data_format in ["csv", "parquet"]:
        _test_for_both(DB_NAME, _test_csv_dataset, data_format)


def _test_csv_dataset(db_type, dataset_location, connection_params, data_format):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location_2 = str(dataset_location) + "_2"
    dataset_dir = Path(__file__).parent / "test_datasets" / f"{db_type}_test_csv_dataset_export_{data_format}"

    shutil.rmtree(dataset_location_2, ignore_errors=True)
    shutil.rmtree(dataset_dir, ignore_errors=True)

    if db_type in ['mysql', 'mariadb']:
        connection_params['database'] += "-2"
        host = connection_params['host']
        user = connection_params['user']
        password = connection_params['password']
        db_name = connection_params['database']
        port = connection_params['port']

        maria_handler = MariaDBHandler(host, user, password, db_name)

        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")

    sdk_2 = AtriumSDK.create_dataset(
        dataset_location=dataset_location_2, database_type=db_type, connection_params=connection_params)

    shutil.rmtree(dataset_dir, ignore_errors=True)
    os.mkdir(dataset_dir)

    write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS, seed=SEED)

    measure_id_list = None
    device_id_list = None
    patient_id_list = None
    start = None
    end = None
    time_units = None
    csv_dur = None

    export_dataset(sdk_1, directory=dataset_dir, device_id_list=device_id_list, patient_id_list=patient_id_list,
                   start=start, end=end, time_units=time_units, csv_dur=csv_dur, measure_id_list=measure_id_list,
                   data_format=data_format, include_scale_factors=True)

    import_dataset(sdk_2, directory=dataset_dir, data_format=data_format)

    assert_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS, seed=SEED)
