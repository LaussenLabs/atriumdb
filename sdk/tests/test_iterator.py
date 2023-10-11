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

    write_mit_bih_to_dataset(sdk, max_records=2, seed=42)
    # Uncomment line below to recreate test files
    # create_test_definition_files(sdk)

    test_parameters = [
        # definition, expected_device_id_type, expected_patient_id_type
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_devices.yaml"), int, type(None)),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_patients.yaml"), type(None), int),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_mrns.yaml"), type(None), int),
        (DatasetDefinition(filename="./example_data/mitbih_seed_42_all_tags.yaml"), int, type(None)),
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

            # Old matrix
            assert isinstance(iterator.get_array_matrix(window_i), np.ndarray)


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

    print()
    print(sdk.get_all_measures())
    print(sdk.get_all_devices())
    print(sdk.get_all_patients())

    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    definition.save("./example_data/mitbih_seed_42_all_devices.yaml", force=True)

    definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)

    definition.save("./example_data/mitbih_seed_42_all_patients.yaml", force=True)

    definition = DatasetDefinition(measures=measures, mrns=mrns)

    definition.save("./example_data/mitbih_seed_42_all_mrns.yaml", force=True)

    definition = DatasetDefinition(measures=measures, device_tags=device_tags)

    definition.save("./example_data/mitbih_seed_42_all_tags.yaml", force=True)
