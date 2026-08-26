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
from atriumdb.windowing.map_definition_sources import map_validated_sources, find_device_patient_data
from tests.testing_framework import _test_for_both

DB_NAME = 'concurrent_device_mappings'


def test_concurrent_device_mappings():
    _test_for_both(DB_NAME, _test_concurrent_device_mappings)


def _test_concurrent_device_mappings(db_type, dataset_location, connection_params):
    # Regression test: a patient concurrently mapped to two devices over the exact
    # same time range (e.g. monitor + ventilator) must yield both
    # (device, patient) tuples from map_validated_sources, not just one.
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    t1 = 1647084000_000_000_000
    t2 = 1647104800_000_000_000

    patient_id = sdk.insert_patient(mrn='123456789')
    monitor_id = sdk.insert_device(device_tag='monitor001')
    vent_id = sdk.insert_device(device_tag='ventilator001')
    sdk.insert_device_patient_data([
        (monitor_id, patient_id, t1, t2),
        (vent_id, patient_id, t1, t2),
    ])

    mapped = map_validated_sources({'patient_ids': {patient_id: [[t1, t2]]}}, sdk)
    tuples = mapped['device_patient_tuples']

    assert (monitor_id, patient_id) in tuples
    assert (vent_id, patient_id) in tuples
    assert tuples[(monitor_id, patient_id)] == [[t1, t2]]
    assert tuples[(vent_id, patient_id)] == [[t1, t2]]

    # The full requested range was covered, so nothing should be left unmapped
    assert 'patient_ids' not in mapped

    # Overlapping (not identical) mappings: ends are out of ascending order when
    # rows are sorted by start, which previously broke the bisect on ends
    patient_id2 = sdk.insert_patient(mrn='987654321')
    monitor2_id = sdk.insert_device(device_tag='monitor002')
    vent2_id = sdk.insert_device(device_tag='ventilator002')
    hour = 3600_000_000_000
    sdk.insert_device_patient_data([
        (monitor2_id, patient_id2, t1, t2),
        (vent2_id, patient_id2, t1 + hour, t1 + 2 * hour),
    ])

    mapped2 = map_validated_sources({'patient_ids': {patient_id2: [[t1, t2]]}}, sdk)
    tuples2 = mapped2['device_patient_tuples']

    assert tuples2[(monitor2_id, patient_id2)] == [[t1, t2]]
    assert tuples2[(vent2_id, patient_id2)] == [[t1 + hour, t1 + 2 * hour]]
    assert 'patient_ids' not in mapped2


def test_find_device_patient_data_overlap():
    # (device_id, patient_id, start, end) rows sorted by start; ends intentionally
    # tie and go out of ascending order
    rows = [
        (1, 10, 0, 100),
        (2, 10, 0, 100),
        (3, 10, 50, 80),
        (4, 10, 200, 300),
    ]
    starts = [r[2] for r in rows]
    ends = [r[3] for r in rows]

    assert find_device_patient_data(rows, starts, ends, 0, 100) == rows[:3]
    assert find_device_patient_data(rows, starts, ends, 90, 250) == [rows[0], rows[1], rows[3]]
    assert find_device_patient_data(rows, starts, ends, 60, 70) == rows[:3]
    assert find_device_patient_data(rows, starts, ends, 110, 150) == []
    assert find_device_patient_data(rows, starts, ends, 400, 500) == []
    assert find_device_patient_data(rows, starts, ends, -50, -10) == []
    assert find_device_patient_data([], [], [], 0, 100) == []
