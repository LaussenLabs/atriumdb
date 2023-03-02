from atriumdb import AtriumSDK
from tests.test_create_dataset import _test_create_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'import_export'


def test_import_export():
    _test_for_both(DB_NAME, _test_create_dataset)


def _test_import_export(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    pass
