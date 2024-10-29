import wfdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from atriumdb import AtriumSDK
import shutil
import os

# Create a new dataset
dataset_location = "./new_dataset"

# WARNING UNCOMMENTING THIS CODE WILL DELETE THE DIRECTORY POINTED AT BY dataset_location
# if os.path.exists(dataset_location):
#     # Reset the local database
#     shutil.rmtree(dataset_location)
sdk = AtriumSDK.create_dataset(dataset_location=dataset_location)

# Define the data source directory
pn_dir = 'mitdb'
record_names = wfdb.get_record_list(pn_dir)

# Insert data into the dataset
for n in tqdm(record_names):
    # Read the record
    record = wfdb.rdrecord(n, pn_dir=pn_dir, return_res=64, physical=False)

    # Get device ID or insert a new device
    device_id = sdk.get_device_id(device_tag=record.record_name)
    if device_id is None:
        device_id = sdk.insert_device(device_tag=record.record_name)

    # Calculate time array in nanoseconds
    freq_nano = record.fs * 1_000_000_000
    time_arr = np.arange(record.sig_len, dtype=np.int64) * int(10 ** 9 // record.fs)

    # Insert labels (annotations)
    annotation = wfdb.rdann(n, 'atr', pn_dir="mitdb", summarize_labels=True, return_label_elements=['description'])
    label_time_idx_array = annotation.sample
    label_time_array = time_arr[label_time_idx_array]
    label_value_list = annotation.description
    # Define list of labels for the record
    labels = []

    # Create labels for each annotation
    for i in range(len(label_value_list)):
        start_time = label_time_array[i]
        end_time = start_time + int(10 ** 9 // record.fs)  # Assuming an annotation lasts for one sample
        labels.append(('Arrhythmia Annotation', device_id, None, None, start_time, end_time))

    sdk.insert_labels(labels=labels, time_units='ns', source_type='device_id')

    # Handle multiple signals in the record
    if record.n_sig > 1:
        for i in range(len(record.sig_name)):
            measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])
            if measure_id is None:
                measure_id = sdk.insert_measure(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])
            scale_m = 1 / record.adc_gain[i]
            scale_b = -record.baseline[i] / record.adc_gain[i]
            sdk.write_data_easy(measure_id, device_id, time_arr, record.d_signal.T[i], freq_nano, scale_m=scale_m, scale_b=scale_b)
    else:
        measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)
        if measure_id is None:
            measure_id = sdk.insert_measure(measure_tag=record.sig_name, freq=freq_nano, units=record.units)
        scale_m = 1 / record.adc_gain
        scale_b = -record.baseline / record.adc_gain
        sdk.write_data_easy(measure_id, device_id, time_arr, record.d_signal, freq_nano, scale_m=scale_m, scale_b=scale_b)

# Survey the dataset
all_measures = sdk.get_all_measures()
all_devices = sdk.get_all_devices()
print("All Measures: ", all_measures)
print("All Devices: ", all_devices)

# Query the data from the dataset and verify
for n in tqdm(record_names):
    record = wfdb.rdrecord(n, pn_dir="mitdb")
    freq_nano = record.fs * 1_000_000_000
    device_id = sdk.get_device_id(device_tag=record.record_name)
    if record.n_sig > 1:
        for i in range(len(record.sig_name)):
            measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])
            _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)
            assert np.allclose(record.p_signal.T[i], read_values)
    else:
        measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)
        _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)
        assert np.allclose(record.p_signal.T, read_values)

# Retrieve labels from the dataset
label_name_dict = sdk.get_all_label_names()
label_names = [label_info['name'] for label_info in label_name_dict.values()]
for record_name in tqdm(record_names):
    label_data = sdk.get_labels(name_list=label_names, device_list=[record_name])
    print(f"Labels for {record_name}: ", label_data)

# Visualize the data
measure_id = 1
device_id = 1
measure_info = sdk.get_measure_info(measure_id=measure_id)
device_info = sdk.get_device_info(device_id=device_id)
freq_nhz = measure_info['freq_nhz']
period_nhz = int((10 ** 18) // freq_nhz)
start_time_n, end_time_n = 0, 1001 * period_nhz
_, times, values = sdk.get_data(measure_id=measure_id, device_id=device_id, start_time_n=start_time_n, end_time_n=end_time_n)

plt.plot(times / (10 ** 9), values)  # convert x-axis units to seconds.
plt.xlabel("Time (Seconds)")
plt.ylabel("Signal Value")
plt.title(f"First 1000 Points of Measure {measure_info['tag']} and Device {device_info['tag']}")
plt.show()


from atriumdb import DatasetDefinition
sdk = AtriumSDK(dataset_location=dataset_location)

# Define a dataset for iteration
# You can either define it using a YAML file or programmatically

# Programmatic definition
measures = [{"tag": measure_info['tag'],
             "freq_nhz": measure_info['freq_nhz'],  # Can specify freq_nhz or freq_hz
             "units": measure_info['unit']}
            for measure_info in sdk.get_all_measures().values()]
device_ids = {device_id: 'all' for device_id in sdk.get_all_devices().keys()}
definition = DatasetDefinition(measures=measures, device_ids=device_ids)

# Iterate over the dataset in windows
window_size = 60
slide_size = 30

# Obtain the iterator
iterator = sdk.get_iterator(definition, window_size, slide_size, time_units="s")

# Now iterate over the data windows
for window_i, window in enumerate(iterator):
    print(f"Window {window_i}")
    print(f"Start Time: {window.start_time}")
    print(f"Device ID: {window.device_id}")
    print(f"Patient ID: {window.patient_id}")

    # Print signals for each measure in the window
    for (measure_tag, measure_freq_nhz, measure_units), signal_dict in window.signals.items():
        print(f"Measure: {measure_tag}, Frequency: {measure_freq_nhz}, Units: {measure_units}")
        print(f"Times: {signal_dict['times']}")
        print(f"Values: {signal_dict['values']}")
        print(f"Expected Count: {signal_dict['expected_count']}, Actual Count: {signal_dict['actual_count']}")
