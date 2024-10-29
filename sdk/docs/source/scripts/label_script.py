from atriumdb import AtriumSDK, DatasetDefinition
import wfdb
from tqdm import tqdm
import numpy as np

import os
import shutil
import csv

label_mapping = {
    'N': 'Normal Sinus',
    'A': 'AFib',
    'O': 'Other',
    '~': 'Noise'
}


def yield_record_data(directory: str):
    reference_file_path = os.path.join(directory, "REFERENCE-v3.csv")

    if not os.path.isfile(reference_file_path):
        raise FileNotFoundError(f"REFERENCE-v3.csv not found in {directory}")

    with open(reference_file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)

        for record_path, label_symbol in reader:
            full_record_path = os.path.join(directory, record_path)

            label_name = label_mapping[label_symbol]
            record_data = wfdb.rdrecord(str(full_record_path), return_res=64, smooth_frames=False,
                                        m2s=True, physical=False)

            yield record_path, label_name, record_data


# Create new AtriumDB dataset
dataset_location = "./challenge_2017_atriumdb"
# WARNING UNCOMMENTING THIS CODE WILL DELETE THE DIRECTORY POINTED AT BY dataset_location
# if os.path.exists(dataset_location):
#     # Reset the local database
#     shutil.rmtree(dataset_location)
sdk = AtriumSDK.create_dataset(dataset_location=dataset_location)


for wfdb_data_dir in ["challenge_2017_data/training/", "challenge_2017_data/validation/"]:
    print(f"Ingesting {wfdb_data_dir} into AtriumDB")
    reference_file_path = os.path.join(wfdb_data_dir, "REFERENCE-v3.csv")
    with open(reference_file_path, mode='r') as csvfile:
        num_rows = len(list(csv.reader(csvfile)))
    for record_path, label_name, record in tqdm(yield_record_data(wfdb_data_dir), total=num_rows):
        device_id = sdk.insert_device(record_path)
        start_frame, end_frame = 0, record.sig_len

        for i, measure_tag in enumerate(record.sig_name):
            # Collect signal data from Record object
            freq_hz = record.fs * record.samps_per_frame[i]
            start_time_s = start_frame / record.fs
            end_time_s = end_frame / record.fs

            gain = record.adc_gain[i]
            baseline = record.baseline[i]
            digital_signal = record.e_d_signal[i]

            # Create/Retrieve measure (signal) ID
            measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")

            # Create timestamp array
            period_s = 1 / freq_hz
            time_data_s = np.arange(digital_signal.size) * period_s + start_time_s

            # Set scale factors
            scale_m = 1 / gain
            scale_b = -baseline / gain

            # Insert Wave Data
            sdk.write_data_easy(measure_id, device_id, time_data_s, digital_signal, freq_hz,
                                scale_m=scale_m, scale_b=scale_b, time_units="s", freq_units="Hz")

            # Insert label
            sdk.insert_label(label_name, start_time_s, end_time_s, device=device_id, time_units="s")


#####################

# Reconnect to AtriumDB Dataset
dataset_location = "./challenge_2017_atriumdb"
sdk = AtriumSDK(dataset_location=dataset_location)

# Generate
measures = []

# Create Dataset Definitions for Train and Test
train_device_ids = None
for wfdb_data_dir in ["challenge_2017_data/training/", "challenge_2017_data/validation/"]:
    pass
