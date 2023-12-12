def transfer_devices(from_sdk, to_sdk, device_id_list=None):
    from_devices = from_sdk.get_all_devices()

    device_map = {}
    for from_device_id, device_info in from_devices.items():
        if device_id_list is None or from_device_id in device_id_list:
            device_tag = device_info['tag']
            device_name = device_info['name']
            to_device_id = to_sdk.insert_device(device_tag=device_tag, device_name=device_name, device_id=from_device_id)

            device_map[from_device_id] = to_device_id

    return device_map
