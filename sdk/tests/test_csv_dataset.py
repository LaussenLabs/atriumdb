from atriumdb import AtriumSDK
import shutil
from pathlib import Path
import os

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from atriumdb.transfer.csv.dataset import export_csv_dataset, import_csv_dataset
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset, assert_partial_mit_bih_to_dataset
from tests.testing_framework import _test_for_both


DB_NAME = 'atrium-csv-transfer'
PARTIAL_DB_NAME = 'atrium-csv-partial-transfer'
MAX_RECORDS = None


def test_csv_dataset():
    _test_for_both(DB_NAME, _test_csv_dataset)


def test_csv_partial_dataset():
    _test_for_both(PARTIAL_DB_NAME, _test_csv_partial_dataset)


def _test_csv_dataset(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location_2 = str(dataset_location) + "_2"
    csv_dataset_dir = Path(__file__).parent / "test_datasets" / f"{db_type}_test_csv_dataset_export"

    shutil.rmtree(dataset_location_2, ignore_errors=True)
    shutil.rmtree(csv_dataset_dir, ignore_errors=True)

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

    shutil.rmtree(csv_dataset_dir, ignore_errors=True)
    os.mkdir(csv_dataset_dir)

    write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS)

    measure_id_list = None
    device_id_list = None
    patient_id_list = None
    start = None
    end = None
    time_units = None
    csv_dur = None

    export_csv_dataset(sdk_1, directory=csv_dataset_dir, measure_id_list=measure_id_list, device_id_list=device_id_list,
                       patient_id_list=patient_id_list, start=start, end=end, time_units=time_units,
                       csv_dur=csv_dur)

    import_csv_dataset(sdk_2, directory=csv_dataset_dir)

    assert_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS)


def _test_csv_partial_dataset(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location_2 = str(dataset_location) + "_partial"
    csv_dataset_dir = Path(__file__).parent / "test_datasets" / f"{db_type}_test_csv_partial_dataset_export"

    shutil.rmtree(dataset_location_2, ignore_errors=True)
    shutil.rmtree(csv_dataset_dir, ignore_errors=True)

    if db_type in ['mysql', 'mariadb']:
        connection_params['database'] += "-partial"
        host = connection_params['host']
        user = connection_params['user']
        password = connection_params['password']
        db_name = connection_params['database']
        port = connection_params['port']

        maria_handler = MariaDBHandler(host, user, password, db_name)

        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")

    sdk_2 = AtriumSDK.create_dataset(
        dataset_location=dataset_location_2, database_type=db_type, connection_params=connection_params)

    shutil.rmtree(csv_dataset_dir, ignore_errors=True)
    os.mkdir(csv_dataset_dir)

    write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS)

    # Customize the measure_id_list and device_id_list to include only specific measure_ids and device_ids
    measure_id_list = [1, 2, 3]
    device_id_list = [1, 2]

    start = 60000000000
    end = 360000000000
    time_units = None
    csv_dur = None

    export_csv_dataset(sdk_1, directory=csv_dataset_dir, measure_id_list=measure_id_list, device_id_list=device_id_list,
                       patient_id_list=None, start=start, end=end, time_units=time_units,
                       csv_dur=csv_dur)

    import_csv_dataset(sdk_2, directory=csv_dataset_dir)

    assert_partial_mit_bih_to_dataset(sdk_2, measure_id_list=measure_id_list, device_id_list=device_id_list,
                                      max_records=MAX_RECORDS, start_nano=start, end_nano=end)

