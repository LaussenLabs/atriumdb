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
import time
import threading
import uvicorn
from atriumdb.atrium_sdk import AtriumSDK
from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both
import pytest

DB_NAME = 'api_test'
MAX_RECORDS = 2
SEED = 42


def test_api():
    def start_server():
        uvicorn.run(app)

    # start server in daemon thread so it exits when complete
    api_thread = threading.Thread(target=start_server, daemon=True)
    api_thread.start()

    # using MITBIH test for normal operation of the sdk
    _test_for_both(DB_NAME, _test_api)

    # test labels functionality of the api
    _test_for_both(DB_NAME, _test_api_labels)


# testing normal operation of the api, getting devices, patients, blocks ect
def _test_api(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # set up the local sdk instance for the api to use
    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    # set up remote mode sdk to connect to the api
    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="http://127.0.0.1:8000", validate_token=False)
    # change the sdk token expiry so the test can work
    api_sdk.token_expiry = time.time() + 1_000_000

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    assert_mit_bih_to_dataset(api_sdk, max_records=MAX_RECORDS, seed=SEED)

    # close api connection
    api_sdk.close()


# test label functionality of the sdk
def _test_api_labels(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # set up the local sdk instance for the api to use
    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    # set up remote mode sdk to connect to the api
    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="http://127.0.0.1:8000", validate_token=False)
    # change the sdk token expiry so the test can work
    api_sdk.token_expiry = time.time() + 1_000_000

    #### test api label functionality ####
    device_tag, device_name = "Monitor A3", "Philips Monitor A3 in Room 2B"
    device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)

    print('Testing get all label names endpoint...')

    # make sure when there are no label names an empty dict is returned
    names_dict = api_sdk.get_all_label_names()
    assert len(names_dict) == 0

    label_names = [
        "Ventricular Flutter",
        "Atrial Fibrillation",
        "Supraventricular Tachycardia",
        "Sinus Bradycardia",
        "Junctional Rhythm",
        "Premature Atrial Contractions (PAC)",
        "Atrioventricular Block (AV Block)",
        "Wolff-Parkinson-White Syndrome",
        "Ventricular Tachycardia",
        "Atrial Flutter"
    ]

    # test to make sure the label names table is empty and a value error is raised
    with pytest.raises(ValueError):
        label_list = api_sdk.get_labels(name_list=label_names)

    # insert the label names
    label_name_ids = [sdk.insert_label_name(name) for name in label_names]

    label_name_dict = api_sdk.get_all_label_names()

    # make sure there are the same ids that were inserted in the dict returned from the api
    assert list(label_name_dict.keys()) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # make sure each label name that was inserted was returned from the api
    for k, v in label_name_dict.items():
        assert v['name'] == label_names[k-1]

    ##### TESTING LABEL SOURCE ENDPOINT #####
    print("Testing label source endpoint...")

    # make sure none is returned when the source doesn't exist
    label_source_id = api_sdk.get_label_source_id(name="Supraventricular Tachycardia")
    assert label_source_id is None

    label_id_inserted = sdk.insert_label_source(name="test")
    # make sure the correct id is returned for the source name
    label_id = api_sdk.get_label_source_id(name="test")
    assert label_id == 1

    # make sure the correct name is returned for the source id
    label_source_info = api_sdk.get_label_source_info(label_source_id=1)
    assert label_source_info['name'] == "test"

    # test to make sure if you supply both or neither parameters an error is returned
    with pytest.raises(ValueError):
        api_sdk._request("GET", "/labels/source", params={'label_source_id': None, 'label_source_name': None})
    with pytest.raises(ValueError):
        api_sdk._request("GET", "/labels/source", params={'label_source_id': 1, 'label_source_name': "test"})


    ##### TESTING LABEL NAME ENDPOINT #####
    print("Testing label name endpoint...")

    # make sure none is returned when the label name doesn't exist
    label_name_id = api_sdk.get_label_name_id(name="test")
    assert label_name_id is None

    # make sure the correct id is returned for the label name
    label_id = api_sdk.get_label_name_id(name="Supraventricular Tachycardia")
    assert label_id == 3

    # make sure the correct name is returned for the label name id
    label_name_info = api_sdk.get_label_name_info(label_name_id=3)
    assert label_name_info['name'] == "Supraventricular Tachycardia"

    # test to make sure if you supply both or neither parameters an error is returned
    with pytest.raises(ValueError):
        api_sdk._request("GET", "/labels/name", params={'label_name_id': None, 'label_name': None})
    with pytest.raises(ValueError):
        api_sdk._request("GET", "/labels/name", params={'label_name_id': 1, 'label_name': "Supraventricular Tachycardia"})

    start_time, end_time = 1000, 2000
    patient_id = sdk.insert_patient(mrn=1234567, first_name="Sterling", middle_name="Malory", last_name="Archer",
                                    dob=283901400000000, weight=83.461, weight_units='kg', height=188, height_units='cm')

    # should raise not implemented error since there is no api mode for insertion
    with pytest.raises(NotImplementedError):
        api_sdk.insert_label(name=label_names[0], patient_id=patient_id, start_time=start_time, end_time=end_time,
                         time_units="ms", label_source='test')

    print("Testing get labels endpoint...")

    # check that a 404 error is returned for a patient with no labels
    with pytest.raises(ValueError):
        labels = api_sdk.get_labels(patient_id_list=[patient_id])
    print("passed 5")

    label_ids, offset = [], 0
    # insert a bunch of labels
    for label_name in label_names:
        for i in range(150):
            label_id = sdk.insert_label(name=label_name, device=device_tag, start_time=start_time+offset, end_time=end_time+offset, time_units="ms", label_source='test')
            label_ids.append(label_id)
            offset += 1000
        offset = 0

    assert len(label_ids) == 1500
    print("passed 6")

    local_labels = sdk.get_labels(name_list=label_names)
    api_labels = api_sdk.get_labels(name_list=label_names)

    assert local_labels == api_labels
    assert len(api_labels) == 1500
    print("passed 7")

    # testing to make sure that no matter what query params are used if they are for the same labels you get the same data
    api_labels = api_sdk.get_labels(device_list=[device_tag])
    assert local_labels == api_labels
    assert len(api_labels) == 1500
    print("passed 8")

    local_labels = sdk.get_labels(name_list=[label_names[0]])
    api_labels = api_sdk.get_labels(name_list=[label_names[0]])

    assert local_labels == api_labels
    assert len(api_labels) == 150
    print("passed 7")

    # Test retrieving labels with both device_list and patient_id_list (should raise an error)
    with pytest.raises(ValueError):
        api_sdk.get_labels(device_list=[device_tag], patient_id_list=[1])

    print("passed 9")

    # Test retrieval of label with non-existent label name (should raise an error)
    with pytest.raises(ValueError):
        api_sdk.get_labels(name_list=["NonExistentLabel"], device_list=[device_tag])

    print("passed 10")

    # Test retrieval of labels using invalid time units
    with pytest.raises(ValueError):
        api_sdk.get_labels(name_list=label_names, device_list=[device_tag], time_units="minutes")

    print("passed 11")

    # Test retrieval of labels using non-existent device tag
    with pytest.raises(ValueError):
        api_sdk.get_labels(name_list=label_names, device_list=["NonExistentDevice"])

    print("passed 12")

    # TEST LABEL HIERARCHY STUFF
    hierarchy_label_dict = \
        {
            "Hierarchy": {
                "Animals": {
                    "Mammals": [
                        "Cats",
                        "Dogs"
                    ],
                    "Birds": [
                        "Parrots",
                        "Sparrows"
                    ]
                },
                "Plants": {
                    "Flowers": [
                        "Roses",
                        "Tulips"
                    ],
                    "Trees": [
                        "Oak",
                        "Pine"
                    ]
                }
            }
        }

    _create_label_hierarchy(sdk, hierarchy_label_dict, parent_id=None)

    # Step 3: Insert a dataset of labels
    labels_data = [
        ('Animals', 101, 1609459200000, 1609462800000, 'Source1'),
        ('Mammals', 102, 1609459300000, 1609462900000, 'Source2'),
        ('Cats', 103, 1609459400000, 1609463000000, 'Source3'),
        ('Dogs', 104, 1609459500000, 1609463100000, 'Source4'),
        ('Birds', 105, 1609459600000, 1609463200000, 'Source5'),
        ('Parrots', 106, 1609459700000, 1609463300000, 'Source6'),
        ('Sparrows', 107, 1609459800000, 1609463400000, 'Source7'),
        ('Plants', 108, 1609459900000, 1609463500000, 'Source8'),
        ('Flowers', 109, 1609460000000, 1609463600000, 'Source9'),
        ('Roses', 110, 1609460100000, 1609463700000, 'Source10'),
        ('Tulips', 111, 1609460200000, 1609463800000, 'Source11'),
        ('Trees', 112, 1609460300000, 1609463900000, 'Source12'),
        ('Oak', 113, 1609460400000, 1609464000000, 'Source13'),
        ('Pine', 114, 1609460500000, 1609464100000, 'Source14')
    ]
    for label_tuple in labels_data:
        device_id = label_tuple[1]
        sdk.insert_device(device_tag=str(device_id), device_id=device_id)

    sdk.insert_labels(labels=labels_data, time_units='ms', source_type='device_id')

    # Step 4: Retrieve and assert labels
    # Retrieve 'Animals' and its descendants
    animals_labels = api_sdk.get_labels(name_list=['Animals'], include_descendants=True)
    print(animals_labels)
    assert any(label['label_name'] == 'Animals' for label in animals_labels)
    assert any(label['label_name'] == 'Mammals' for label in animals_labels)
    assert any(label['label_name'] == 'Cats' for label in animals_labels)
    assert any(label['label_name'] == 'Dogs' for label in animals_labels)
    assert any(label['label_name'] == 'Birds' for label in animals_labels)
    assert any(label['label_name'] == 'Parrots' for label in animals_labels)
    assert any(label['label_name'] == 'Sparrows' for label in animals_labels)

    # check for types
    assert isinstance(animals_labels[0]['label_entry_id'], int)
    assert isinstance(animals_labels[0]['label_name_id'], int)
    assert isinstance(animals_labels[0]['label_name'], str)
    assert isinstance(animals_labels[0]['requested_name_id'], int)
    assert isinstance(animals_labels[0]['requested_name'], str)
    assert isinstance(animals_labels[0]['device_id'], int)
    assert isinstance(animals_labels[0]['device_tag'], str)
    assert isinstance(animals_labels[0]['start_time_n'], int)
    assert isinstance(animals_labels[0]['end_time_n'], int)
    assert isinstance(animals_labels[0]['label_source_id'], int)
    assert isinstance(animals_labels[0]['label_source'], str)

    # Retrieve 'Plants' and its descendants
    plants_labels = api_sdk.get_labels(name_list=['Plants'], include_descendants=True)
    assert any(label['label_name'] == 'Plants' for label in plants_labels)
    assert any(label['label_name'] == 'Flowers' for label in plants_labels)
    assert any(label['label_name'] == 'Roses' for label in plants_labels)
    assert any(label['label_name'] == 'Tulips' for label in plants_labels)
    assert any(label['label_name'] == 'Trees' for label in plants_labels)
    assert any(label['label_name'] == 'Oak' for label in plants_labels)
    assert any(label['label_name'] == 'Pine' for label in plants_labels)

    print("Testing children endpoint...")
    children_by_name = api_sdk.get_label_name_children(name="Animals")
    for child in children_by_name:
        assert child['name'] in ['Mammals', 'Birds']

    print("Testing parent endpoint...")
    parent_by_name = api_sdk.get_label_name_parent(name="Oak")
    assert parent_by_name['name'] == "Trees"

    returned_hierarchy = api_sdk.get_all_label_name_descendents(name="Hierarchy")
    assert returned_hierarchy['Animals'] == {'Birds': {'Parrots': {}, 'Sparrows': {}}, 'Mammals': {'Cats': {}, 'Dogs': {}}}
    assert returned_hierarchy["Plants"]["Trees"] == {'Oak': {}, 'Pine': {}}

    # close api connection
    api_sdk.close()


def _create_label_hierarchy(sdk: AtriumSDK, hierarchy: dict, parent_id=None):
    """
    Recursively creates labels based on a hierarchical structure.

    :param sdk: An instance of the AtriumSDK.
    :param hierarchy: A dictionary representing the hierarchical structure of labels.
    :param parent_id: The ID of the parent label. None for top-level labels.
    """
    for label, children in hierarchy.items():
        # Insert the current label and get its ID
        label_id = sdk.insert_label_name(name=label, parent=parent_id)

        # If the current label has children, recursively create them
        if isinstance(children, dict):
            # Recursive call for a sub-dictionary
            _create_label_hierarchy(sdk, children, parent_id=label_id)
        elif isinstance(children, list):
            # Iterate over the list of child labels
            for child_label in children:
                sdk.insert_label_name(name=child_label, parent=label_id)