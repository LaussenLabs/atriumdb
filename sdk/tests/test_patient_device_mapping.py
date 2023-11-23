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

import pytest

DB_NAME = 'test_device_patient_mapping'


def test_device_patient_mapping():
    _test_for_both(DB_NAME, _test_device_patient_mapping)


def _test_device_patient_mapping(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Time boundaries for testing
    t1 = 1647084000_000_000_000
    t2 = 1647104800_000_000_000

    # Test case 1: Single Mapping Match
    patient_id1 = sdk.insert_patient(mrn='123456789')
    device_id1 = sdk.insert_device(device_tag='monitor001')
    sdk.insert_device_patient_data([(device_id1, patient_id1, t1, t2)])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id1) == device_id1
    assert sdk.convert_device_to_patient_id(t1, t2, device_id1) == patient_id1

    # Test case 2: Single Mapping Overlap
    patient_id2 = sdk.insert_patient(mrn='123456780')
    device_id2 = sdk.insert_device(device_tag='monitor002')
    sdk.insert_device_patient_data([(device_id2, patient_id2, t1 - 3600_000_000_000, t2 + 3600_000_000_000)])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id2) == device_id2
    assert sdk.convert_device_to_patient_id(t1, t2, device_id2) == patient_id2

    # Test case 3: Out-of-Range Before
    patient_id3 = sdk.insert_patient(mrn='123456781')
    device_id3 = sdk.insert_device(device_tag='monitor003')
    sdk.insert_device_patient_data([(device_id3, patient_id3, t1 + 7200_000_000_000, t2 + 7200_000_000_000)])

    assert sdk.convert_patient_to_device_id(t1 - 7200_000_000_000, t1 - 3600_000_000_000, patient_id3) is None
    assert sdk.convert_device_to_patient_id(t1 - 7200_000_000_000, t1 - 3600_000_000_000, device_id3) is None

    # Test case 4: Out-of-Range After
    patient_id4 = sdk.insert_patient(mrn='123456782')
    device_id4 = sdk.insert_device(device_tag='monitor004')
    sdk.insert_device_patient_data([(device_id4, patient_id4, t1, t2)])

    assert sdk.convert_patient_to_device_id(t2 + 3600_000_000_000, t2 + 7200_000_000_000, patient_id4) is None
    assert sdk.convert_device_to_patient_id(t2 + 3600_000_000_000, t2 + 7200_000_000_000, device_id4) is None

    # Test case 5: Matching Edge To Edge
    patient_id5 = sdk.insert_patient(mrn='123456783')
    device_id5 = sdk.insert_device(device_tag='monitor005')
    sdk.insert_device_patient_data([
        (device_id5, patient_id5, t1, t1 + 3600_000_000_000),
        (device_id5, patient_id5, t1 + 3600_000_000_000, t2)
    ])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id5) == device_id5
    assert sdk.convert_device_to_patient_id(t1, t2, device_id5) == patient_id5

    # Test case 6: Multiple Overlapping
    patient_id6 = sdk.insert_patient(mrn='123456784')
    device_id6 = sdk.insert_device(device_tag='monitor006')
    sdk.insert_device_patient_data([
        (device_id6, patient_id6, t1, t1 + 7200_000_000_000),
        (device_id6, patient_id6, t1 + 3600_000_000_000, t2)
    ])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id6) == device_id6
    assert sdk.convert_device_to_patient_id(t1, t2, device_id6) == patient_id6

    # Test case 7: Disjoint Mappings
    patient_id7 = sdk.insert_patient(mrn='123456785')
    device_id7 = sdk.insert_device(device_tag='monitor007')
    sdk.insert_device_patient_data([
        (device_id7, patient_id7, t1, t1 + 1800_000_000_000),
        (device_id7, patient_id7, t2 - 1800_000_000_000, t2)
    ])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id7) is None
    assert sdk.convert_device_to_patient_id(t1, t2, device_id7) is None

    # Test case 8: Multiple Patients
    patient_id8a = sdk.insert_patient(mrn='123456786')
    patient_id8b = sdk.insert_patient(mrn='123456787')
    device_id8 = sdk.insert_device(device_tag='monitor008')
    sdk.insert_device_patient_data([
        (device_id8, patient_id8a, t1, t2),
        (device_id8, patient_id8b, t1, t2)
    ])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id8a) == device_id8
    with pytest.raises(ValueError):
        sdk.convert_device_to_patient_id(t1, t2, device_id8)

    # Test case 9: Multiple Devices
    patient_id9 = sdk.insert_patient(mrn='123456788')
    device_id9a = sdk.insert_device(device_tag='monitor009a')
    device_id9b = sdk.insert_device(device_tag='monitor009b')
    sdk.insert_device_patient_data([
        (device_id9a, patient_id9, t1, t1 + 3600_000_000_000),
        (device_id9b, patient_id9, t2 - 3600_000_000_000, t2)
    ])

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id9) is None
    assert sdk.convert_device_to_patient_id(t1, t2, 'monitor009a') is None

    # Test case 10: Invalid Input
    try:
        sdk.convert_patient_to_device_id('invalid', 'input', patient_id=patient_id1)
    except ValueError as e:
        assert str(e) == "start_time and end_time must be integers."

    try:
        sdk.convert_device_to_patient_id('invalid', 'input', device_id1)
    except ValueError as e:
        assert str(e) == "start_time and end_time must be integers."

    # Test case 11: No Mappings
    patient_id11 = sdk.insert_patient(mrn='1234567890')
    device_id11 = sdk.insert_device(device_tag='monitor011')

    assert sdk.convert_patient_to_device_id(t1, t2, patient_id11) is None
    assert sdk.convert_device_to_patient_id(t1, t2, device_id11) is None
