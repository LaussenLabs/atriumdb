import numpy as np
import wfdb  # pip install wfdb
from atriumdb import AtriumSDK
import shutil
import os

# Step 1: Create or Connect to a Dataset

# Option 1: Create a new local dataset using SQLite
dataset_location = "./new_dataset"
if os.path.exists(dataset_location):
    # Reset the local database
    shutil.rmtree(dataset_location)
sdk = AtriumSDK.create_dataset(dataset_location=dataset_location)

# Option 2: Create a new local dataset using MariaDB
# connection_params = {
#     'host': "localhost",
#     'user': "user",
#     'password': "pass",
#     'database': "new_dataset",
#     'port': 3306
# }
# sdk = AtriumSDK.create_dataset(dataset_location=dataset_location, database_type="mysql", connection_params=connection_params)

# Option 3: Connect to an existing SQLite dataset
# sdk = AtriumSDK(dataset_location=dataset_location)

# Option 4: Connect to a dataset using MariaDB
# connection_params = {
#     'host': "localhost",
#     'user': "user",
#     'password': "pass",
#     'database': "new_dataset",
#     'port': 3306
# }
# sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type="mysql", connection_params=connection_params)

# Option 5: Connect to a remote dataset using the API
# api_url = "http://example.com/v1"
# token = "4e78a93749ead7893"
# sdk = AtriumSDK(api_url=api_url, token=token, metadata_connection_type="api")


# Step 2: Pull example data from WFDB (MITDB database)

# Fetch record 100 from MITDB
record_name, pn_dir = "100", "mitdb"

record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=False, physical=False)
segments = record.segments if isinstance(record, wfdb.MultiRecord) else [record]

# Define a new source (device) in the database
device_tag = "MITDB_record_100"
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
        time_data_s = np.arange(digital_signal.size) * period_s + start_time_s

        # Create/Retrieve measure (signal) ID
        measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")

        # Set scale factors
        scale_m = 1 / gain
        scale_b = -baseline / gain

        sdk.write_data_easy(measure_id, device_id, time_data_s, digital_signal, freq_hz,
                            scale_m=scale_m, scale_b=scale_b, time_units="s", freq_units="Hz")


# Repull the analog (physical) MITBIH Record
record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=True, physical=True)


for i, measure_tag in enumerate(record.sig_name):
    analog_signal = record.e_p_signal[i]
    freq_hz = record.fs * record.samps_per_frame[i]
    measure_id = sdk.get_measure_id(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")
    # Read the data back from the database
    _, read_time_data, read_value_data = sdk.get_data(
        measure_id=measure_id,
        start_time_n=0,
        end_time_n=max_end_time,
        device_id=device_id,
        time_units="s",
    )

    # Verify the read data matches the WFDB Record
    assert read_value_data.shape == analog_signal.shape
    assert np.allclose(read_value_data, analog_signal)

print("Data read and verified successfully!")
