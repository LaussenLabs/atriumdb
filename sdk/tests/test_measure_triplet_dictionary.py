import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-measure-triplet'


def test_measure_triplet():
    _test_for_both(DB_NAME, _test_measure_triplet)


def _test_measure_triplet(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id_dict = {}  # Will store the inserted measure_ids
    measure_triplet_list = []  # Will store the triplets for retrieval
    freq_unit_keys = {"nHz": 1, "uHz": 10 ** 3, "mHz": 10 ** 6, "Hz": 10 ** 9, "kHz": 10 ** 12, "MHz": 10 ** 15}

    for tag in ["1", "2", "3"]:
        for freq_unit, multiplier in freq_unit_keys.items():
            freq = 1
            freq_nhz = freq * multiplier
            triplet = (tag, freq_nhz, "mV")
            measure_id = sdk.insert_measure(tag, freq, "mV", freq_units=freq_unit)
            measure_id_dict[triplet] = measure_id
            measure_triplet_list.append(triplet)

    # Test getting inserted triplets
    result_dict = sdk.get_measure_triplet_to_id_dictionary(measure_triplet_list)
    assert result_dict == measure_id_dict

    # Test ValueError for non-existing triplets
    with pytest.raises(ValueError):
        sdk.get_measure_triplet_to_id_dictionary([("tag_does_not_exist", 1, "mV")])

    # Test single match for (tag, None, None)
    measure_id = sdk.insert_measure("4", 200, "mV")
    result = sdk.get_measure_triplet_to_id_dictionary([("4", None, None)])
    assert result == {("4", None, None): measure_id}

    # Test ValueError for multiple matches
    sdk.insert_measure("5", 300, "mV")
    sdk.insert_measure("5", 400, "mV")
    with pytest.raises(ValueError):
        sdk.get_measure_triplet_to_id_dictionary([("5", None, None)])


