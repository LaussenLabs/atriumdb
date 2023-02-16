from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'create_db'


def test_create_dataset():
    _test_for_both(DB_NAME, _test_create_dataset)


def _test_create_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
