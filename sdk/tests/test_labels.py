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

import pytest
import numpy as np

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'labels'


def test_labels():
    _test_for_both(DB_NAME, _test_labels)


def _test_labels(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Test insertion of a new device and get the ID
    device_tag = "Monitor A3"
    device_name = "Philips Monitor A3 in Room 2B"
    device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)
    assert device_id is not None, "Failed to insert a new device or retrieve its ID"

    new_device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)
    assert new_device_id == device_id, "Device ID mismatch for the same device tag"

    label_name = "Running"
    label_id = sdk.get_label_name_id(name=label_name)
    assert label_id is None, "There's a label where there shouldn't be"

    start_time = 1000
    end_time = 2000
    sdk.insert_label(name=label_name, device=device_tag, start_time=start_time, end_time=end_time, time_units="ms")
    label_id = sdk.get_label_name_id(name=label_name)
    assert label_id is not None, "Failed to insert a new label type or retrieve its ID"

    label_set_info = sdk.get_label_name_info(label_name_id=label_id)
    assert label_set_info is not None, "Failed to retrieve label set info"
    assert label_set_info["id"] == label_id, "Mismatch in label set ID"
    assert label_set_info["name"] == label_name, "Mismatch in label set name"

    labels = sdk.get_labels(name_list=[label_name], device_list=[device_tag])
    assert len(labels) == 1, "Incorrect number of labels retrieved"
    retrieved_label = labels[0]
    assert retrieved_label['label_name_id'] == label_id, "Label ID mismatch"
    assert retrieved_label['device_id'] == device_id, "Device ID mismatch"

    label_2_name = "Idle"
    label_2_start_time = 2000
    label_2_end_time = 3000
    labels_to_insert = [
        (label_2_name, device_tag, label_2_start_time, label_2_end_time, None)
    ]
    sdk.insert_labels(labels=labels_to_insert, time_units="ms", source_type="device_tag")

    # Retrieve multiple labels and verify their values
    retrieved_labels = sdk.get_labels(name_list=[label_name, label_2_name], device_list=[device_tag])
    assert len(retrieved_labels) == 2, "Incorrect number of labels retrieved"

    # Test insertion of a label with a non-existent device tag
    with pytest.raises(ValueError):
        sdk.insert_label(name=label_name, device="NonExistentDevice", start_time=start_time, end_time=end_time)

    # Test insertion of a label with an invalid time unit
    with pytest.raises(ValueError):
        sdk.insert_label(name=label_name, device=device_tag, start_time=start_time, end_time=end_time,
                         time_units="minutes")

    # Test retrieving labels with both device_list and patient_id_list (should raise an error)
    with pytest.raises(ValueError):
        sdk.get_labels(device_list=[device_tag], patient_id_list=[1])

    # Test retrieval of label with non-existent label name (should raise an error)
    with pytest.raises(ValueError):
        sdk.get_labels(name_list=["NonExistentLabel"], device_list=[device_tag])

    # Test retrieval of labels using invalid time units
    with pytest.raises(ValueError):
        sdk.get_labels(name_list=[label_name], device_list=[device_tag], time_units="minutes")

    # Test retrieval of labels using non-existent device tag
    with pytest.raises(ValueError):
        sdk.get_labels(name_list=[label_name], device_list=["NonExistentDevice"])

    # Testing get_label_time_series
    expected_series_1 = np.array([1, 1, 1])
    label_series = sdk.get_label_time_series(label_name=label_name, device_tag=device_tag,
                                             start_time=start_time, end_time=end_time + 500, sample_period=500,
                                             time_units="ms")
    assert np.array_equal(label_series, expected_series_1), "Mismatch in expected time series data"

    with pytest.raises(ValueError):
        sdk.get_label_time_series(label_name=label_name, label_name_id=label_id, device_tag=device_tag,
                                  start_time=start_time, end_time=end_time, sample_period=500, time_units="ms")

    with pytest.raises(ValueError):
        sdk.get_label_time_series(label_name=label_name, device_id=device_id, device_tag=device_tag,
                                  start_time=start_time, end_time=end_time, sample_period=500, time_units="ms")

    with pytest.raises(ValueError):
        sdk.get_label_time_series(label_name=label_name, device_tag=device_tag,
                                  start_time=start_time, end_time=end_time, sample_period=500, time_units="minutes")

    timestamp_array = np.array([1000, 1500, 2000, 2500, 3000], dtype=np.int64)
    label_series_2 = sdk.get_label_time_series(label_name=label_name, device_tag=device_tag,
                                               timestamp_array=timestamp_array, time_units="ms")

    expected_series_2 = np.array([1, 1, 1, 0, 0])
    assert np.array_equal(label_series_2, expected_series_2), "Mismatch in expected time series data using timestamp array"

    with pytest.raises(ValueError):
        sdk.get_label_time_series(label_name=label_name, device_tag=device_tag, start_time=start_time,
                                  sample_period=500, time_units="ms")

    with pytest.raises(ValueError):
        sdk.get_label_time_series(label_name=label_name, device_tag=device_tag, time_units="ms")