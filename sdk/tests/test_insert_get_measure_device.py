from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'measure_device'


def test_insert_get_measure_device():
    _test_for_both(DB_NAME, _test_insert_get_measure_device)


def _test_insert_get_measure_device(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    pass
