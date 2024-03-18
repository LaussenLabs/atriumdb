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


def transfer_devices(from_sdk, to_sdk, device_id_list=None):
    from_devices = from_sdk.get_all_devices()

    device_map = {}
    for from_device_id, device_info in from_devices.items():
        if device_id_list is None or from_device_id in device_id_list:
            device_tag = device_info['tag']
            device_name = device_info['name']

            # Check if device_id already exists
            check_device_info = to_sdk.get_device_info(from_device_id)
            if check_device_info is not None:
                # if its the same device, return the id without inserting
                if check_device_info['tag'] == device_tag:
                    to_device_id = from_device_id
                    device_map[from_device_id] = to_device_id
                    continue
                else:
                    # The device_id is taken but its a different device so ask for a new id when inserting
                    from_device_id = None
                    
            to_device_id = to_sdk.insert_device(device_tag=device_tag, device_name=device_name, device_id=from_device_id)

            device_map[from_device_id] = to_device_id

    return device_map
