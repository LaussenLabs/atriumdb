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

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.windowing.definition_splitter import get_label_duration_list
from tests.testing_framework import _test_for_both

DB_NAME = 'concurrent_device_mappings'

S = 10 ** 9


def test_concurrent_device_mappings():
    _test_for_both(DB_NAME, _test_concurrent_device_mappings)


def _test_concurrent_device_mappings(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure("hr", 1, "bpm", freq_units="Hz")
    dev_1, dev_2, dev_3, dev_4 = (sdk.insert_device(f"dev{i}") for i in range(1, 5))
    patient_1, patient_2, patient_3 = (sdk.insert_patient() for _ in range(3))

    for device_id, start_s, end_s in [(dev_3, 0, 50), (dev_1, 100, 1000), (dev_2, 300, 400), (dev_4, 0, 100)]:
        sdk.write_segment(measure_id, device_id, np.arange(end_s - start_s), start_s * S, freq=1, freq_units="Hz")

    # Patient 1 is on dev_2 during a dev_1 stay, and patients 2 and 3 share dev_4.
    sdk.insert_device_patient_data([
        (dev_3, patient_1, 0, 50 * S),
        (dev_1, patient_1, 100 * S, 1000 * S),
        (dev_2, patient_1, 300 * S, 400 * S),
        (dev_4, patient_2, 0, 100 * S),
        (dev_4, patient_3, 0, 100 * S),
    ])

    definition = DatasetDefinition(measures=["hr"], patient_ids={patient_1: [{"start": 100 * S, "end": 500 * S}]})
    definition.validate(sdk=sdk)
    sources = definition.validated_data_dict["sources"]
    assert sources["device_patient_tuples"] == {
        (dev_1, patient_1): [[100 * S, 500 * S]],
        (dev_2, patient_1): [[300 * S, 400 * S]],
    }

    definition = DatasetDefinition(measures=["hr"], device_ids={dev_4: "all"})
    definition.validate(sdk=sdk)
    assert set(definition.validated_data_dict["sources"]["device_patient_tuples"]) == {
        (dev_4, patient_2), (dev_4, patient_3)}

    # Nested labels on dev_1 are both found, and each is counted only inside [100, 500).
    sdk.insert_label("event", 100 * S, 900 * S, device=dev_1)
    sdk.insert_label("event", 300 * S, 400 * S, device=dev_1)
    label_duration_list, _ = get_label_duration_list(sources, [sdk.get_label_name_id("event")], sdk)
    assert label_duration_list[0][1] == patient_1
    assert label_duration_list[0][4] == 500 * S
