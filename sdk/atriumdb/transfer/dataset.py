from typing import List

from atriumdb import AtriumSDK
from atriumdb.adb_functions import convert_value_to_nanoseconds

MIN_TRANSFER_TIME = -(2**63)
MAX_TRANSFER_TIME = (2**63) - 1


def transfer_data(from_sdk: AtriumSDK, to_sdk: AtriumSDK, measure_id_list: List[int] = None,
                  device_id_list: List[int] = None, patient_id_list: List[int] = None, mrn_list: List[int] = None,
                  start: int = None, end: int = None, time_units: str = None, batch_size=None):

    batch_size = 40 if batch_size is None else batch_size
    time_units = 'ns' if time_units is None else time_units

    if start is not None:
        start = convert_value_to_nanoseconds(start, time_units)

    if end is not None:
        end = convert_value_to_nanoseconds(end, time_units)

    if len(measure_id_list) == 0:
        raise ValueError("Must specify at least one measure.")

    # Transfer measures
    transfer_measures(from_sdk, to_sdk, measure_id_list=measure_id_list)

    if patient_id_list is not None or mrn_list is not None:
        device_patient_list = from_sdk.get_device_patient_data(
            device_id_list=device_id_list, patient_id_list=patient_id_list, mrn_list=mrn_list,
            start_time=start, end_time=end)

        device_id_list = list(set([row[0] for row in device_patient_list]))

        # Transfer devices
        transfer_devices(from_sdk, to_sdk, device_id_list=device_id_list)

        # Transfer device_patients
        to_sdk.insert_device_patient_data(device_patient_data=device_patient_list)

        for measure_id in from_sdk.get_all_measures().keys():
            if measure_id_list is not None and measure_id not in measure_id_list:
                continue

            for device_id, patient_id, start_time, end_time in device_patient_list:
                headers, times, values = from_sdk.get_data(
                    measure_id, start_time, end_time, device_id=device_id, analog=False)

                if len(headers) == 0:
                    continue

                ingest_data(to_sdk, measure_id, device_id, headers, times, values)

    else:
        # Transfer devices
        transfer_devices(from_sdk, to_sdk, device_id_list=device_id_list)

        for measure_id in from_sdk.get_all_measures().keys():
            if measure_id_list is not None and measure_id not in measure_id_list:
                continue

            for device_id in from_sdk.get_all_devices().keys():
                if device_id_list is not None and device_id not in device_id_list:
                    continue

                block_list = from_sdk.get_block_id_list(int(measure_id), start_time_n=start,
                                                        end_time_n=end, device_id=device_id)

                if len(block_list) == 0:
                    continue

                file_id_list = list(set([row[3] for row in block_list]))
                filename_dict = from_sdk.get_filename_dict(file_id_list)

                start_block = 0
                while start_block < len(block_list):
                    block_batch = block_list[start_block:start_block+batch_size]
                    headers, times, values = from_sdk.get_data_from_blocks(block_batch, filename_dict, measure_id,
                                                                           MIN_TRANSFER_TIME, MAX_TRANSFER_TIME)

                    ingest_data(to_sdk, measure_id, device_id, headers, times, values)
                    start_block += batch_size


def ingest_data(to_sdk, measure_id, device_id, headers, times, values):
    if all([h.scale_m == headers[0].scale_m for h in headers]) and \
            all([h.scale_b == headers[0].scale_b for h in headers]) and \
            all([h.freq_nhz == headers[0].freq_nhz for h in headers]):
        to_sdk.write_data_easy(measure_id, device_id, time_data=times, value_data=values,
                               freq=headers[0].freq_nhz, scale_m=headers[0].scale_m,
                               scale_b=headers[0].scale_b)
    else:
        val_index = 0
        for h in headers:
            to_sdk.write_data_easy(measure_id, device_id, time_data=times[val_index:val_index + h.num_vals],
                                   value_data=values[val_index:val_index + h.num_vals], freq=h.freq_nhz,
                                   scale_m=h.scale_m, scale_b=h.scale_b)
            val_index += h.num_vals


def transfer_measures(from_sdk, to_sdk, measure_id_list=None):
    from_measures = from_sdk.get_all_measures()
    for measure_id, measure_info in from_measures.items():
        if measure_id_list is None or measure_id in measure_id_list:
            measure_tag = measure_info['tag']
            freq = measure_info['freq_nhz']
            units = measure_info['unit']
            measure_name = measure_info['name']
            to_sdk.insert_measure(measure_tag=measure_tag, freq=freq, units=units, measure_name=measure_name)


def transfer_devices(from_sdk, to_sdk, device_id_list=None):
    from_devices = from_sdk.get_all_devices()

    for device_id, device_info in from_devices.items():
        if device_id_list is None or device_id in device_id_list:
            device_tag = device_info['tag']
            device_name = device_info['name']
            to_sdk.insert_device(device_tag=device_tag, device_name=device_name)
