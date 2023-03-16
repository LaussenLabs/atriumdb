from atriumdb import AtriumSDK
import pandas as pd
from typing import Union, Dict
from pathlib import PurePath


def import_csv_to_sdk(sdk: AtriumSDK, filename: Union[str, PurePath], metadata: Dict[str, Union[str, float, int]]):
    df = pd.read_csv(filename)

    device_tag = metadata['device_tag']
    measure_tag = metadata['measure_tag']
    freq_nhz = metadata['freq']
    units = metadata['unit']
    scale_m = metadata.get('scale_m', None)
    scale_b = metadata.get('scale_b', None)

    period_ns = (10 ** 18) // freq_nhz
    times, values = parse_csv_data(df)
    assert times.size > 0, f"No data in file {filename}."

    measure_id, device_id = get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units)
    assert measure_id is not None, f"Measure {measure_tag} {freq_nhz} {units} could not be inserted."
    assert device_id is not None, f"Device {device_tag} could not be inserted."

    sdk.write_data_easy(measure_id, device_id, times, values, freq_nhz, scale_m=scale_m, scale_b=scale_b)

    return measure_id, device_id, int(times[0]), int(times[-1] + period_ns)


def parse_csv_column_info(df):
    device_tag, measure_tag, freq_nhz, units, scale_m, scale_b = None, None, None, None, None, None

    # Extract the column names from the dataframe
    col_names = df.columns

    time_col_name = col_names[0]
    val_col_name = col_names[1]

    device_tag = time_col_name.split(" ")[0]

    val_col_split = val_col_name.split(" ")

    measure_tag = val_col_split[0]
    freq_nhz = int(val_col_split[1])
    units = val_col_split[2]
    scale_m = float(val_col_split[3]) if len(val_col_split) > 3 else None
    scale_b = float(val_col_split[4]) if len(val_col_split) > 4 else None

    return device_tag, measure_tag, freq_nhz, units, scale_m, scale_b


def parse_csv_data(df):
    times = df[df.columns[0]].to_numpy()
    values = df[df.columns[1]].to_numpy()

    return times, values


def get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units):
    measure_id = sdk.get_measure_id(measure_tag, freq_nhz, units)
    if measure_id is None:
        measure_id = sdk.insert_measure(measure_tag, freq_nhz, units)

    device_id = sdk.get_device_id(device_tag)
    if device_id is None:
        device_id = sdk.insert_device(device_tag)

    return measure_id, device_id
