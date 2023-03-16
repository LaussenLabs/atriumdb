from atriumdb import AtriumSDK
import shutil

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset, assert_partial_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-transfer'
PARTIAL_DB_NAME = 'partial-atrium-transfer'


def test_transfer():
    _test_for_both(DB_NAME, _test_transfer)


def test_partial_transfer():
    _test_for_both(PARTIAL_DB_NAME, _test_partial_transfer)


def _test_transfer(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location = str(dataset_location) + "_2"
    shutil.rmtree(dataset_location, ignore_errors=True)

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
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk_1)

    measure_id_list = None
    device_id_list = None
    patient_id_list = None
    start = None
    end = None
    time_units = None
    batch_size = None

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, measure_id_list=measure_id_list,
                  device_id_list=device_id_list, patient_id_list=patient_id_list, start=start, end=end,
                  time_units=time_units, batch_size=batch_size)

    assert_mit_bih_to_dataset(sdk_2)


def _test_partial_transfer(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location = str(dataset_location) + "_partial"
    shutil.rmtree(dataset_location, ignore_errors=True)

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
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk_1)

    # Customize the measure_id_list and device_id_list to include only specific measure_ids and device_ids
    measure_id_list = [1, 2, 3]
    device_id_list = [1, 2]

    start = 60000000000
    end = 360000000000
    time_units = None
    batch_size = None

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, measure_id_list=measure_id_list,
                  device_id_list=device_id_list, patient_id_list=None, start=start, end=end,
                  time_units=time_units, batch_size=batch_size)

    assert_partial_mit_bih_to_dataset(sdk_2, measure_id_list, device_id_list, start_nano=start, end_nano=end)
