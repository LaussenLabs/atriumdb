from typing import List

from atriumdb import AtriumSDK
from atriumdb.adb_functions import convert_value_to_nanoseconds


def transfer_data(from_sdk: AtriumSDK, to_sdk: AtriumSDK, measure_id_list: List[int] = None,
                  device_id_list: List[int] = None, patient_id_list: List[int] = None, mrn_list: List[int] = None,
                  start: int = None, end: int = None, time_units: str = None):
    time_units = 'ns' if time_units is None else time_units

    if start is not None:
        start = convert_value_to_nanoseconds(start, time_units)

    if end is not None:
        end = convert_value_to_nanoseconds(end, time_units)

    if len(measure_id_list) == 0:
        raise ValueError("Must specify at least one measure.")

    from_devices = from_sdk.get_all_devices()
    from_measures = from_sdk.get_all_measures()

    # Transfer measures
    for measure_id, measure_info in from_measures.items():
        if measure_id_list is None or measure_id in measure_id_list:
            to_sdk.insert_measure(measure_info)

    if patient_id_list is not None or mrn_list is not None:
        device_patient_list = from_sdk.get_device_patient_data(
            device_id_list=device_id_list, patient_id_list=patient_id_list, mrn_list=mrn_list,
            start_time=start, end_time=end)

        for device_id, patient_id, start_time, end_time in device_patient_list:
            pass
