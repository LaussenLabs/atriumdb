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
import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from tests.test_mit_bih import write_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'iterator'


def test_iterator():
    _test_for_both(DB_NAME, _test_iterator)


def _test_iterator(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # larger test
    write_mit_bih_to_dataset(sdk, max_records=2, seed=42)
    # Uncomment line below to recreate test files
    # create_test_definition_files(sdk)

    test_parameters = [
        # definition, expected_device_id_type, expected_patient_id_type
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_devices.yaml"), int, int),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_patients.yaml"), int, int),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_mrns.yaml"), int, int),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_tags.yaml"), int, int),
    ]

    window_size_nano = 1_024 * 1_000_000_000
    for definition, expected_device_id_type, expected_patient_id_type in test_parameters:
        iterator = sdk.get_iterator(definition, window_size_nano, window_size_nano, num_windows_prefetch=None)

        for window_i, window in enumerate(iterator):
            assert isinstance(window.start_time, int)
            assert isinstance(window.device_id, expected_device_id_type)
            assert isinstance(window.patient_id, expected_patient_id_type)

            for (measure_tag, measure_freq_nhz, measure_units), signal_dict in window.signals.items():
                assert isinstance(signal_dict['times'], np.ndarray)
                assert isinstance(signal_dict['values'], np.ndarray)

            # Labels
            assert isinstance(window.label_time_series, np.ndarray)
            assert isinstance(window.label, np.ndarray)

    # Check for the case of partial windows
    partial_freq_nano = 1_000_000_000
    partial_period_nano = (10 ** 18) // partial_freq_nano
    partial_device_id = sdk.insert_device(device_tag="partial_device")
    partial_measure_id = sdk.insert_measure(measure_tag="partial_measure", freq=partial_freq_nano, units="mV")

    start_time = 1_000_000_000
    num_values = 100
    end_time = start_time + (num_values * partial_period_nano)
    times = np.arange(start_time, end_time, partial_period_nano)
    values = (np.sin(times) * 1000).astype(np.int64)
    scale_m = 1 / 1000
    scale_b = 0

    sdk.write_data_easy(
        partial_measure_id, partial_device_id, times, values, partial_freq_nano, scale_m=scale_m, scale_b=scale_b)

    # Add a patient
    patient_id = sdk.insert_patient()

    # Only map half the data
    half_time = int(times[(num_values // 2) + 1])
    sdk.insert_device_patient_data([(partial_device_id, patient_id, start_time, half_time)])

    # get definition
    definition = DatasetDefinition(measures=["partial_measure"], device_ids={partial_device_id: "all"})

    window_size_nano = partial_period_nano * 25
    iterator = sdk.get_iterator(definition, window_size_nano, window_size_nano, num_windows_prefetch=None)

    for window_i, window in enumerate(iterator):
        for (measure_tag, measure_freq_nhz, measure_units), signal_dict in window.signals.items():
            first_nan_idx = get_index_of_first_nan(signal_dict['values'])
            first_nan_time = int(signal_dict['times'][first_nan_idx])
            if first_nan_time - partial_period_nano < half_time:
                assert window.patient_id == patient_id
            else:
                assert window.patient_id is None

        assert window.label_time_series is None
        assert window.label is None


def create_test_definition_files(sdk):
    measures = []
    for measure_id, measure_info in sdk.get_all_measures().items():
        tag = measure_info['tag']
        freq_nhz = measure_info['freq_nhz']
        units = measure_info['unit']
        measures.append({'tag': tag, 'freq_nhz': freq_nhz, 'units': units})

    device_ids = {device_id: "all" for device_id in sdk.get_all_devices().keys()}
    patient_ids = {patient_id: "all" for patient_id in sdk.get_all_patients().keys()}
    mrns = {patient_info['mrn']: "all" for patient_info in sdk.get_all_patients().values()}
    device_tags = {device_info['tag']: "all" for device_info in sdk.get_all_devices().values()}

    labels = [label_info['name'] for label_info in sdk.get_all_label_names().values()]

    print()
    print(sdk.get_all_measures())
    print(sdk.get_all_devices())
    print(sdk.get_all_patients())
    print(sdk.get_all_label_names())

    definition = DatasetDefinition(measures=measures, device_ids=device_ids, labels=labels)

    definition.save("./example_data/mitbih_seed_42_all_devices.yaml", force=True)

    definition = DatasetDefinition(measures=measures, patient_ids=patient_ids, labels=labels)

    definition.save("./example_data/mitbih_seed_42_all_patients.yaml", force=True)

    definition = DatasetDefinition(measures=measures, mrns=mrns, labels=labels)

    definition.save("./example_data/mitbih_seed_42_all_mrns.yaml", force=True)

    definition = DatasetDefinition(measures=measures, device_tags=device_tags, labels=labels)

    definition.save("./example_data/mitbih_seed_42_all_tags.yaml", force=True)


def get_index_of_first_nan(arr):
    nan_index = np.argmax(np.isnan(arr))
    if nan_index == 0 and not np.isnan(arr[0]):
        return len(arr) - 1
    return nan_index
