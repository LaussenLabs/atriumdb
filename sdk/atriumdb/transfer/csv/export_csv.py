from pathlib import PurePath, Path
from typing import Union

from atriumdb import AtriumSDK
import pandas as pd

from atriumdb.transfer.csv.json_file_metadata import create_json_metadata_dict


def export_csv_from_sdk(sdk: AtriumSDK, filename: Union[str, PurePath], measure_id, start_time, end_time,
                        device_id=None, include_scale_factors=False):
    headers, times, values = sdk.get_data(measure_id, start_time, end_time, device_id=device_id, analog=False)

    if len(headers) == 0:
        return None

    device_tag, measure_tag, freq_nhz, units = get_source_info(sdk, measure_id, device_id)
    scale_m, scale_b = None, None

    if include_scale_factors:
        scale_m, scale_b = get_scale_factors(headers)

    df = create_dataframe(measure_tag, times, values)

    df.to_csv(filename, index=False)

    metadata = create_json_metadata_dict(
        measure_tag, freq_nhz, units, device_tag, start_time, end_time, scale_m, scale_b, None)
    filename = Path(filename)
    filename = str(filename.name)

    return {filename: metadata}, df


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


def create_dataframe(measure_tag, times, values):
    data = {f"Epoch Nano": times, f"{measure_tag}": values}
    df = pd.DataFrame(data)

    return df
