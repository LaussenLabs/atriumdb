from atriumdb import AtriumSDK
import pandas as pd


def import_csv(sdk: AtriumSDK, filename: str):
    df = pd.read_csv(filename)

    device_tag, measure_tag, freq_nhz, units, scale_m, scale_b = parse_csv_column_info(df)
    times, values = parse_csv_data(df)

    measure_id, device_id = get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units, scale_m, scale_b)

    sdk.write_data_easy(measure_id, device_id, times, values, freq_nhz, scale_m=scale_m, scale_b=scale_b)


def parse_csv_column_info(df):
    device_tag, measure_tag, freq_nhz, units, scale_m, scale_b = None, None, None, None, None, None

    # Extract the column names from the dataframe
    col_names = df.columns

    # Parse the device_tag and measure_tag
    for col_name in col_names:
        if "device_tag" in col_name.lower():
            device_tag = col_name.split(" ")[0]
        elif "measure_tag" in col_name.lower():
            measure_tag = col_name.split(" ")[0]
            freq_nhz = col_name.split(" ")[2]
            units = col_name.split(" ")[3]
            scale_m = col_name.split(" ")[4] if len(col_name.split(" ")) > 4 else None
            scale_b = col_name.split(" ")[5] if len(col_name.split(" ")) > 5 else None

    return device_tag, measure_tag, freq_nhz, units, scale_m, scale_b


def parse_csv_data(df):
    times = df[df.columns[0]].to_numpy()
    values = df[df.columns[1]].to_numpy()

    return times, values


def get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units, scale_m, scale_b):
    measure_id = sdk.get_measure_id(measure_tag, freq_nhz, units)
    device_id = sdk.get_device_id(device_tag)

    return measure_id, device_id

