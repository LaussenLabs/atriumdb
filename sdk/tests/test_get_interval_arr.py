from atriumdb import AtriumSDK

from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-interval'

MAX_RECORDS = 1


def test_get_interval_arr():
    _test_for_both(DB_NAME, _test_get_interval_arr)


def _test_get_interval_arr(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    pass
