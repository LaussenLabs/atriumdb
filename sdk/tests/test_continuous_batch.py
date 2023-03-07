import numpy as np

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'continuous_batch'


def test_continuous_batch():
    _test_for_both(DB_NAME, _test_continuous_batch)


def _test_continuous_batch(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    value_data = np.arange(1024 * 3)

