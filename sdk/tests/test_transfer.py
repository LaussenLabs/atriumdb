from atriumdb import AtriumSDK
import shutil

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset, assert_partial_mit_bih_to_dataset
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-transfer'
PARTIAL_DB_NAME = 'partial-atrium-transfer'


def test_transfer():
    # _test_for_both(DB_NAME, _test_transfer)
    # _test_for_both(PARTIAL_DB_NAME, _test_partial_transfer)
    _test_for_both(DB_NAME, _test_transfer_with_patient_context)
    # _test_for_both(DB_NAME, _test_transfer_with_patient_context_deidentify_timeshift)


def _test_partial_transfer(db_type, dataset_location, connection_params):
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    dataset_location = str(dataset_location) + "_partial"
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    write_mit_bih_to_dataset(sdk_1)

    # Customize the measure_id_list and device_id_list to include only specific measure_ids and device_ids
    measure_id_list = [1, 2, 3]
    device_id_list = [1, 2]

    start = 60000000000
    end = 360000000000
    time_units = None
    batch_size = None

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, measure_id_list=measure_id_list, device_id_list=device_id_list,
                  patient_id_list=None, start=start, end=end, time_units=time_units, batch_size=batch_size)

    assert_partial_mit_bih_to_dataset(sdk_2, measure_id_list, device_id_list, start_nano=start, end_nano=end)


def _test_transfer(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Test

    device_patient_dict = write_mit_bih_to_dataset(sdk_1)

    measure_id_list = None
    device_id_list = None
    patient_id_list = None
    start = None
    end = None
    time_units = None
    batch_size = None

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, measure_id_list=measure_id_list, device_id_list=device_id_list,
                  patient_id_list=patient_id_list, start=start, end=end, time_units=time_units, batch_size=batch_size)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict)


def _test_transfer_with_patient_context(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Test
    device_patient_dict = write_mit_bih_to_dataset(sdk_1)

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, include_patient_context=True)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, use_patient_id=True)


def _test_transfer_with_patient_context_deidentify_timeshift(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    # Test
    device_patient_dict = write_mit_bih_to_dataset(sdk_1)

    transfer_data(from_sdk=sdk_1, to_sdk=sdk_2, include_patient_context=True, deidentify=True, time_shift=500)

    assert_mit_bih_to_dataset(sdk_2, device_patient_map=device_patient_dict, deidentify=True, time_shift=500)
