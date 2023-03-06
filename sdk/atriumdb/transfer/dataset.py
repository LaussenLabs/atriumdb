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

    device_patient_list = from_sdk.get_device_patient_data(
        device_id_list=device_id_list, patient_id_list=patient_id_list, mrn_list=mrn_list,
        start_time=start, end_time=end)
