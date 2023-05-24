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
from atriumdb.atrium_sdk import convert_to_nanohz
from tests.testing_framework import _test_for_both

DB_NAME = 'get_measure_device'


def test_insert_get_measure_device():
    _test_for_both(DB_NAME, _test_insert_get_measure_device)


def _test_insert_get_measure_device(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # test the insert_measure method
    measure_tag = "ECG Lead II - 500 Hz"
    freq_hz = 500
    freq_units = "Hz"
    measure_name = "Electrocardiogram Lead II Configuration 500 Hertz"
    units = "mV"
    sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, freq_units=freq_units, measure_name=measure_name,
                       units=units)
    measure_id = sdk.get_measure_id(measure_tag, freq_hz, units, freq_units)
    assert measure_id is not None

    # test the get_measure_info method
    measure_info = sdk.get_measure_info(measure_id)
    assert measure_info is not None
    assert measure_info['id'] == measure_id
    assert measure_info['tag'] == measure_tag
    assert measure_info['name'] == measure_name
    assert measure_info['freq_nhz'] == convert_to_nanohz(freq_hz, freq_units)
    assert measure_info['unit'] == units

    # test the get_measure_info method when the requested measure hasn't been inserted
    measure_info = sdk.get_measure_info(100)
    assert measure_info is None

    # test the insert_device method
    device_tag = "Monitor A3"
    device_name = "Philips Monitor A3 in Room 2B"
    sdk.insert_device(device_tag=device_tag, device_name=device_name)
    device_id = sdk.get_device_id(device_tag)
    assert device_id is not None

    # test the get_device_info method
    device_info = sdk.get_device_info(device_id)
    assert device_info is not None
    assert device_info['id'] == device_id
    assert device_info['tag'] == device_tag
    assert device_info['name'] == device_name

    # test the get_device_info method when the requested device hasn't been inserted
    device_info = sdk.get_device_info(100)
    assert device_info is None

    measure_id = sdk.get_measure_id("non_existent_measure_tag", 100, "units")
    assert measure_id is None

    device_id = sdk.get_device_id("non_existent_device_tag")
    assert device_id is None
