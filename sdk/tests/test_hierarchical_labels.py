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

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'hierarchical_labels'

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


def test_hierarchical_labels():
    _test_for_both(DB_NAME, _test_hierarchical_labels)


def _test_hierarchical_labels(db_type, dataset_location, connection_params):
    # Step 1: Create the dataset
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Step 2: Insert the hierarchical labels
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
    animals_labels = sdk.get_labels(name_list=['Animals'], include_descendants=True)
    assert any(label['label_name'] == 'Animals' for label in animals_labels)
    assert any(label['label_name'] == 'Mammals' for label in animals_labels)
    assert any(label['label_name'] == 'Cats' for label in animals_labels)
    assert any(label['label_name'] == 'Dogs' for label in animals_labels)
    assert any(label['label_name'] == 'Birds' for label in animals_labels)
    assert any(label['label_name'] == 'Parrots' for label in animals_labels)
    assert any(label['label_name'] == 'Sparrows' for label in animals_labels)

    # Retrieve 'Plants' and its descendants
    plants_labels = sdk.get_labels(name_list=['Plants'], include_descendants=True)
    assert any(label['label_name'] == 'Plants' for label in plants_labels)
    assert any(label['label_name'] == 'Flowers' for label in plants_labels)
    assert any(label['label_name'] == 'Roses' for label in plants_labels)
    assert any(label['label_name'] == 'Tulips' for label in plants_labels)
    assert any(label['label_name'] == 'Trees' for label in plants_labels)
    assert any(label['label_name'] == 'Oak' for label in plants_labels)
    assert any(label['label_name'] == 'Pine' for label in plants_labels)

    children_by_name = sdk.get_label_name_children(name="Animals")
    for child in children_by_name:
        print(child)
        assert child['name'] in ['Mammals', 'Birds']

    parent_by_name = sdk.get_label_name_parent(name="Oak")
    print(parent_by_name)
    assert parent_by_name['name'] == "Trees"

    print()
    print(sdk.get_all_label_name_descendents(name="Animals"))

    print()
    print(sdk.get_all_label_name_descendents(name="Hierarchy"))


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
