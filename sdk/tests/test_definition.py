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
import os
from pathlib import Path

from atriumdb import AtriumSDK, DatasetDefinition
from tests.testing_framework import _test_for_both

DB_NAME = 'definition'

# Get the directory where this test file is located
TEST_DIR = Path(__file__).parent
EXAMPLE_DATA_DIR = TEST_DIR / "example_data"


@pytest.mark.parametrize(
    "filename, expected_exception, expected_warning, expected_message",
    [
        ("error1.yaml", ValueError, None, "Unexpected key: patient_id"),
        ("error2.yaml", ValueError, None, "Patient ID John must be an integer"),
        ("error4.yaml", ValueError, None, "Invalid time key: en. Allowed keys are: "
                                          "start, end, time0, pre, post"),
        ("error6.yaml", ValueError, None, "patient_id 12345: start time 1682739300000000000 must be "
                                          "less than end time 1682739300000000000"),
        ("error7.yaml", ValueError, None, "pre cannot be negative"),
        ("error8.yaml", ValueError, None, "Device ID device_1 must be an integer"),
        ("error9.yaml", ValueError, None, "'pre' and 'post' cannot be provided without 'time0'"),
        ("error10.yaml", ValueError, None, "MRN must be convertible to an integer"),
        ("error11.yaml", ValueError, None, "Duplicate measure found: tag_1"),
        ("correct1.yaml", None, None, None),
        ("correct2.yaml", None, None, None),
        ("correct3.yaml", None, None, None),
        ("mitbih_seed_42_all_devices.yaml", None, None, None),
        ("mitbih_seed_42_all_patients.yaml", None, None, None),
        ("mitbih_seed_42_all_mrns.yaml", None, None, None),
        ("mitbih_seed_42_all_tags.yaml", None, None, None),
    ],
)
def test_definition_file_formatting(filename, expected_exception, expected_warning, expected_message):
    # Construct the full path to the file
    full_path = EXAMPLE_DATA_DIR / filename

    if expected_exception is None and expected_warning is None:
        try:
            DatasetDefinition(filename=str(full_path))
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
    elif expected_exception is not None:
        with pytest.raises(expected_exception, match=expected_message):
            DatasetDefinition(filename=str(full_path))
    else:
        with pytest.warns(expected_warning, match=expected_message):
            DatasetDefinition(filename=str(full_path))


def test_advanced_definition():
    _test_for_both(DB_NAME, _test_advanced_definition)


def _test_advanced_definition(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    highest_number = 10_000
    repetitions_per_number = 20
    repeated_data = np.repeat(np.arange(highest_number), repetitions_per_number)

    measure_id = sdk.insert_measure("example_measure", 1, "units", freq_units="Hz")
    device_id = sdk.insert_device("example_device")

    sdk.write_segment(measure_id, device_id, repeated_data, 0, freq=1, time_units="s", freq_units="Hz",
                      scale_m=1.0, scale_b=0.0)

    measures = ['example_measure']
    device_ids = {device_id: "all"}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    definition.validate(sdk)

    filter_fn = lambda window: not np.any(
        np.round(window.signals[("example_measure", 1.0, "units")]['values']) % 2 == 0)
    definition.filter(sdk, filter_fn=filter_fn, window_duration=repetitions_per_number,
                      window_slide=repetitions_per_number, time_units="s", allow_partial_windows=True)

    filtered_data = []
    for window in sdk.get_iterator(definition, window_duration=repetitions_per_number,
                                   window_slide=repetitions_per_number, time_units="s"):
        filtered_data.append(window.signals[("example_measure", 1.0, "units")]['values'])

    result_data = np.concatenate(filtered_data)
    result_data = result_data[~np.isnan(result_data)]

    floating_repeated_data = (repeated_data * 1.0) + 0.0
    floating_odd_numbers = floating_repeated_data[repeated_data % 2 == 1]

    assert np.allclose(floating_odd_numbers, result_data)


def test_filter_slide_greater_than_window():
    """Test filtering with slide size greater than window size to verify warning behavior."""
    _test_for_both(DB_NAME, _test_filter_slide_greater_than_window)


def _test_filter_slide_greater_than_window(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Create test data
    test_data = np.arange(1000)
    measure_id = sdk.insert_measure("test_measure", 10, "units", freq_units="Hz")
    device_id = sdk.insert_device("test_device")

    sdk.write_segment(measure_id, device_id, test_data, 0, freq=10, time_units="s", freq_units="Hz",
                      scale_m=1.0, scale_b=0.0)

    # Create definition
    measures = ['test_measure']
    device_ids = {device_id: "all"}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    # Validate the definition
    definition.validate(sdk)

    # Test filter with slide > window (should trigger warning)
    filter_fn = lambda window: True  # Accept all windows
    window_duration = 5  # 5 seconds
    window_slide = 10  # 10 seconds (greater than window_duration)

    # This should generate a UserWarning about high memory usage
    with pytest.warns(UserWarning, match=r"Window size .* and slide .* are not equal.*high memory usage"):
        definition.filter(
            sdk,
            filter_fn=filter_fn,
            window_duration=window_duration,
            window_slide=window_slide,
            time_units="s",
            allow_partial_windows=True
        )

    # Verify that the filter operation completed successfully
    assert definition.filtered_window_size == window_duration * 1_000_000_000  # converted to nanoseconds
    assert definition.filtered_window_slide == window_slide * 1_000_000_000  # converted to nanoseconds

    # Test that we can still iterate through the filtered data
    window_count = 0
    for window in sdk.get_iterator(definition, window_duration=window_duration,
                                   window_slide=window_slide, time_units="s"):
        window_count += 1
        # Verify window structure
        assert ("test_measure", 10.0, "units") in window.signals

    # Should have some windows since we accepted all of them
    assert window_count > 0