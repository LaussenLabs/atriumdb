import shutil
from pathlib import Path
import random

import numpy as np
import time
from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, create_gap_arr_fast, merge_gap_data
from tests.test_mit_bih import create_gaps


def test_simple():
    # Set up the AtriumSDK to create a new dataset
    sqlite_dataset_path = Path(__file__).parent / "test_datasets" / f"sqlite_simple_dataset"
    shutil.rmtree(sqlite_dataset_path, ignore_errors=True)
    sqlite_dataset_path.unlink(missing_ok=True)
    sdk = AtriumSDK.create_dataset(dataset_location=sqlite_dataset_path)

    freq_hz = 1  # frequency of the sine wave in Hz
    sampling_rate_hz = (10 ** 9)  # sampling rate in samples per second (1 sample per nanosecond)
    freq_nano = int(freq_hz * (10 ** 9))

    period_ns = (10 ** 18) // freq_nano

    # Create timestamps for the sine wave
    num_values = 1_000_000

    for start_time_ns in np.arange(3, dtype=np.int64) * (10 ** 16):
        # sdk.block.block_size = random.choice([2 ** exp for exp in range(11, 21)])
        # print(f"block size {sdk.block.block_size}")
        gap_total = 0
        gap_data_2d = create_gaps(num_values, period_ns)
        times = np.arange(num_values, dtype=np.int64) * (10 ** 9) + start_time_ns
        for index, duration in gap_data_2d:
            times[index:] += duration
            gap_total += duration

        # Generate the sine wave
        t = np.linspace(0, times.size, num=times.size, endpoint=False) / sampling_rate_hz

        scale_m = None
        scale_b = None
        value_data = np.sin(2 * np.pi * freq_hz * t)
        scale_m_inverse = float(10 ** 10)
        scale_b_inverse = 0.0

        scale_m = 1 / scale_m_inverse
        scale_b = -scale_b_inverse

        value_data += scale_b_inverse
        value_data *= scale_m_inverse
        value_data = value_data.astype(np.int64)

        expected_value_floats = (value_data.astype(np.float64) * scale_m) + scale_b

        # Define signal and source in AtriumDB
        sig_name = "sine_wave"
        device_tag = "synthetic_generator"
        new_measure_id = sdk.insert_measure(measure_tag=sig_name, freq=freq_hz, freq_units="Hz")
        new_device_id = sdk.insert_device(device_tag=device_tag)

        # Write the generated sine wave data into AtriumDB

        # Detect Value Type
        if np.issubdtype(value_data.dtype, np.integer):
            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        # Decide Time Type
        raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
        if random.random() < 0.5:
            # Time Value pair encoding
            encoded_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
            sdk.block.t_compression = 5
            sdk.block.t_compression_level = 2
            sdk.block.v_compression = 5
            sdk.block.v_compression_level = 2
            # sdk.block.t_compression = 3
            # sdk.block.t_compression_level = 12
        else:
            # Gap Array encoding
            encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Call the write_data method with the determined parameters
        sdk.write_data(new_measure_id, new_device_id, times, value_data, freq_nano, int(times[0]),
                       raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                       encoded_value_type=encoded_v_t, scale_m=scale_m, scale_b=scale_b)

        sdk.block.t_compression = 1
        sdk.block.t_compression_level = 0

        # Read the data back from AtriumDB to verify
        start_time_nano = int(times[0])
        end_time_nano = int(times[-1]) + period_ns
        _, read_time_data, read_value_data = sdk.get_data(measure_id=new_measure_id, start_time_n=start_time_nano,
                                                          end_time_n=end_time_nano, device_id=new_device_id)

        # Verifying the written and read data
        assert np.array_equal(times, read_time_data)
        assert np.allclose(expected_value_floats, read_value_data)
