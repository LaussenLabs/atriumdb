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
    """Map every selected source device onto a destination device, creating it if needed.

    Returns ``{source_device_id: destination_device_id}``. The source id is ALWAYS the
    key, including when the destination had to allocate a different id -- callers look
    devices up by source id (``extract_device_ids``, the label transfer) and a missing or
    mis-keyed entry silently sends that device's data to ``device_id=None``.
    """
    from_devices = from_sdk.get_all_devices()

    device_map = {}
    for src_device_id, device_info in from_devices.items():
        if device_id_list is None or src_device_id in device_id_list:
            device_tag = device_info['tag']
            device_name = device_info['name']

            # Ask for the source's own id by default, so ids line up across datasets
            # where they can.
            requested_device_id = src_device_id

            # Check if device_id already exists
            check_device_info = to_sdk.get_device_info(src_device_id)
            if check_device_info is not None:
                # if its the same device, return the id without inserting
                if check_device_info['tag'] == device_tag:
                    device_map[src_device_id] = src_device_id
                    continue
                else:
                    # The device_id is taken but its a different device so ask for a new
                    # id when inserting. Only the REQUESTED id becomes None here -- the
                    # source id stays intact as the map key. The loop variable must not be
                    # overwritten, or this device's entry would use None and later lookups
                    # would resolve to
                    # None: its data was written to device_id=None and the label transfer
                    # raised KeyError.
                    requested_device_id = None

            to_device_id = to_sdk.insert_device(device_tag=device_tag, device_name=device_name,
                                                device_id=requested_device_id)

            device_map[src_device_id] = to_device_id

    return device_map
