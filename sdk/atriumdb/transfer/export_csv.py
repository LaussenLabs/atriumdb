from pathlib import PurePath
from typing import Union

from atriumdb import AtriumSDK
import pandas as pd


def export_csv_from_sdk(sdk: AtriumSDK, filename: Union[str, PurePath], measure_id, start_time, end_time, device_id=None):
    headers, times, values = sdk.get_data(measure_id, start_time, end_time, device_id=device_id)
    assert len(headers) > 0, f"No Data found for measure_id: {measure_id}, start_time: {start_time}, " \
                             f"end_time: {end_time}, device_id: {device_id}"

    device_tag, measure_tag, freq_nhz, units = get_source_info(sdk, measure_id, device_id)
    scale_m, scale_b = get_scale_factors(headers)

    df = create_dataframe(device_tag, measure_tag, freq_nhz, units, scale_m, scale_b, times, values)

    df.to_csv(filename, index=False)


def get_source_info(sdk, measure_id, device_id):
    measure_info = sdk.get_measure_info(measure_id)
    device_info = sdk.get_device_info(device_id)

    assert measure_info is not None, f"No measure id {measure_id} found"
    assert device_info is not None, f"No device_ id {device_id} found"

    measure_tag = measure_info['tag']
    freq_nhz = measure_info['freq_nhz']
    units = measure_info['unit']
    device_tag = device_info['tag']

    return device_tag, measure_tag, freq_nhz, units


def get_scale_factors(headers):
    scale_m = headers[0].scale_m
    scale_b = headers[0].scale_b

    return scale_m, scale_b


def create_dataframe(device_tag, measure_tag, freq_nhz, units, scale_m, scale_b, times, values):
    col_name = f"{measure_tag} {freq_nhz} {units}"
    if scale_m is not None and scale_b is not None:
        col_name += f" {scale_m} {scale_b}"

    data = {f"{device_tag} Epoch Nano": times, col_name: values}
    df = pd.DataFrame(data)

    return df
