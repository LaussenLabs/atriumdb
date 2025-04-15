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

from atriumdb import AtriumSDK, DatasetDefinition
from tests.testing_framework import _test_for_both

DB_NAME = 'definition'


@pytest.mark.parametrize(
    "filename, expected_exception, expected_warning, expected_message",
    [
        ("./example_data/error1.yaml", ValueError, None, "Unexpected key: patient_id"),
        ("./example_data/error2.yaml", ValueError, None, "Patient ID John must be an integer"),
        ("./example_data/error3.yaml", None, UserWarning, "patient_id 12345: The epoch for start: 1659344515 looks "
                                                          "like it's formatted in seconds. However start will be "
                                                          "interpreted as nanosecond data."),
        ("./example_data/error4.yaml", ValueError, None, "Invalid time key: en. Allowed keys are: "
                                                         "start, end, time0, pre, post"),
        ("./example_data/error5.yaml", None, UserWarning, "patient_id 12345: The epoch for pre: 60 looks like it's "
                                                          "formatted in seconds. However pre will be interpreted "
                                                          "as nanosecond data."),
        ("./example_data/error6.yaml", ValueError, None, "patient_id 12345: start time 1682739300000000000 must be "
                                                         "less than end time 1682739300000000000"),
        ("./example_data/error7.yaml", ValueError, None, "pre cannot be negative"),
        ("./example_data/error8.yaml", ValueError, None, "Device ID device_1 must be an integer"),
        ("./example_data/error9.yaml", ValueError, None, "'pre' and 'post' cannot be provided without 'time0'"),
        ("./example_data/error10.yaml", ValueError, None, "MRN must be convertible to an integer"),
        ("./example_data/error11.yaml", ValueError, None, "Duplicate measure found: tag_1"),
        ("./example_data/correct1.yaml", None, None, None),
        ("./example_data/correct2.yaml", None, None, None),
        ("./example_data/correct3.yaml", None, None, None),
        ("./example_data/mitbih_seed_42_all_devices.yaml", None, None, None),
        ("./example_data/mitbih_seed_42_all_patients.yaml", None, None, None),
        ("./example_data/mitbih_seed_42_all_mrns.yaml", None, None, None),
        ("./example_data/mitbih_seed_42_all_tags.yaml", None, None, None),
    ],
)
def test_definition_file_formatting(filename, expected_exception, expected_warning, expected_message):
    if expected_exception is None and expected_warning is None:
        try:
            DatasetDefinition(filename=filename)
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
    elif expected_exception is not None:
        with pytest.raises(expected_exception, match=expected_message):
            DatasetDefinition(filename=filename)
    else:
        with pytest.warns(expected_warning, match=expected_message):
            DatasetDefinition(filename=filename)


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

    filter_fn = lambda window: not np.any(np.round(window.signals[("example_measure", 1.0, "units")]['values']) % 2 == 0)
    definition.filter(sdk, filter_fn=filter_fn, window_duration=repetitions_per_number, window_slide=repetitions_per_number, time_units="s", allow_partial_windows=True)

    filtered_data = []
    for window in sdk.get_iterator(definition, window_duration=repetitions_per_number,
                                   window_slide=repetitions_per_number, time_units="s"):
        filtered_data.append(window.signals[("example_measure", 1.0, "units")]['values'])

    result_data = np.concatenate(filtered_data)
    result_data = result_data[~np.isnan(result_data)]

    floating_repeated_data = (repeated_data * 1.0) + 0.0
    floating_odd_numbers = floating_repeated_data[repeated_data % 2 == 1]

    assert np.allclose(floating_odd_numbers, result_data)
