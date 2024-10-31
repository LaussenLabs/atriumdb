import numpy as np
import wfdb
from atriumdb import AtriumSDK
import shutil
import os
from tqdm import tqdm

dataset_location = "./new_dataset"
# WARNING UNCOMMENTING THIS CODE WILL DELETE THE DIRECTORY POINTED AT BY dataset_location
# if os.path.exists(dataset_location):
#     # Reset the local database
#     shutil.rmtree(dataset_location)
sdk = AtriumSDK.create_dataset(dataset_location=dataset_location)

# Fetch all records from MITDB
pn_dir = "mitdb"
record_names = wfdb.get_record_list(pn_dir)

# record_names = record_names[:5]

for record_name in tqdm(record_names):
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=False, physical=False)
    segments = record.segments if isinstance(record, wfdb.MultiRecord) else [record]

    # Define a new source (device) in the database
    device_tag = f"MITDB_record_{record_name}"
    device_id = sdk.insert_device(device_tag=device_tag)

    end_frame = 0
    max_end_time = 0
    for segment in segments:
        start_frame = end_frame
        end_frame += segment.sig_len
        if not segment or segment.sig_len == 0:
            continue
        for i, measure_tag in enumerate(segment.sig_name):
            # Collect signal data from Record object
            freq_hz = segment.fs * segment.samps_per_frame[i]
            start_time_s = start_frame / segment.fs
            end_time_s = end_frame / segment.fs
            max_end_time = max(max_end_time, end_time_s)
            gain = segment.adc_gain[i]
            baseline = segment.baseline[i]
            digital_signal = segment.e_d_signal[i]

            # Create timestamp array
            period_s = 1 / freq_hz

            # Create/Retrieve measure (signal) ID
            measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")

            # Set scale factors
            scale_m = 1 / gain
            scale_b = -baseline / gain

            sdk.write_message(measure_id, device_id, digital_signal, start_time_s,
                              scale_m=scale_m, scale_b=scale_b)

for record_name in tqdm(record_names):
    # Repull the analog (physical) MITBIH Record
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=True, physical=True)
    device_tag = f"MITDB_record_{record_name}"
    device_id = sdk.get_device_id(device_tag=device_tag)

    for i, measure_tag in enumerate(record.sig_name):
        analog_signal = record.e_p_signal[i]
        freq_hz = record.fs * record.samps_per_frame[i]
        measure_id = sdk.get_measure_id(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")
        # Read the data back from the database
        _, read_time_data, read_value_data = sdk.get_data(
            measure_id=measure_id,
            start_time_n=0,
            end_time_n=record.sig_len / freq_hz,
            device_id=device_id,
            time_units="s",
        )

        # Verify the read data matches the WFDB Record
        assert read_value_data.shape == analog_signal.shape
        if not np.allclose(read_value_data, analog_signal):
            print(f"Data mismatch in record {record_name}, measure {measure_tag}")
        else:
            print(f"Data read and verified successfully for record {record_name}, measure {measure_tag}!")

print("All records processed and verified successfully!")
