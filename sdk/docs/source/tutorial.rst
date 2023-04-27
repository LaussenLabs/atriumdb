Creating a Dataset, Inserting Data, and Querying Data - A Comprehensive Tutorial
################################################################################

In this tutorial, we will walk you through the process of creating a dataset, inserting new data, and querying that data using the AtriumSDK library. We will use the provided example to read data from the MIT-BIH Arrhythmia Database and store it in our dataset.

Prerequisites
-------------

- Python 3.8 or higher
- AtriumSDK library
- wfdb library

You can install the required libraries using pip:

.. code-block:: bash

   pip install atriumdb wfdb

Creating a New Dataset
----------------------

Creating a New Dataset
----------------------

First, let's create a new dataset using the AtriumSDK library. We will use the default SQLite metadata database for simplicity. The `create_dataset` method allows you to specify various options such as the type of metadata database to use, the protection mode, and the behavior when new data overlaps with existing data.

.. code-block:: python

   from atriumdb import AtriumSDK

   sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset")

You can also create a dataset with a different metadata database, such as MariaDB or MySQL, by providing the `database_type` and `connection_params` parameters. For example:

.. code-block:: python

   connection_params = {
       'host': "localhost",
       'user': "user",
       'password': "pass",
       'database': "new_dataset",
       'port': 3306
   }
   sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", connection_params=connection_params)

Additionally, you can set the protection mode and overwrite behavior using the `protected_mode` and `overwrite` parameters. For example, to create a dataset with protection mode enabled and an overwrite behavior set to "error":

.. code-block:: python

   sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", protected_mode=True, overwrite="error")

These additional configurations allow you to customize the dataset according to your needs and preferences.

Inserting Data into the Dataset
--------------------------------

Now that we have created a new dataset, let's insert some data into it. We will use the provided example to read data from the MIT-BIH Arrhythmia Database and store it in our dataset.

.. code-block:: python

   import wfdb
   from pathlib import Path

   DEFAULT_WFDB_DATA_DIR = Path(__file__).parent / 'wfdb_data'
   DEFAULT_DATASET_NAME = 'mitdb'

   def get_records(dataset_name=None):
       dataset_name = DEFAULT_DATASET_NAME if dataset_name is None else dataset_name
       dataset_dir_path = DEFAULT_WFDB_DATA_DIR / dataset_name

       if not dataset_dir_path.is_dir():
           dataset_dir_path.mkdir(parents=True, exist_ok=True)
           wfdb.dl_database(dataset_name, str(dataset_dir_path))

       for record_name in wfdb.get_record_list(dataset_name):
           record = wfdb.rdrecord(str(dataset_dir_path / record_name))
           yield record

Now we need to define the measures, devices, and patients in our dataset. For our example, we will define a single measure for the ECG signal, a single device for the MIT-BIH Arrhythmia Database, and a single patient.

.. code-block:: python

   # Define a new signal
   freq = 360
   freq_units = "Hz"
   measure_tag = "ECG"
   measure_name = "Electrocardiogram"
   units = "mV"

   # Insert the new signal into the dataset
   new_measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units, measure_name=measure_name)

   # Define a new data source
   device_tag = "MIT-BIH"
   device_name = "MIT-BIH Arrhythmia Database"

   # Insert the new data source into the dataset
   new_device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)

   # Define a new patient record
   patient_id = 1
   mrn = "123456"
   gender = "M"
   dob = 946684800000000000
   first_name = "John"
   middle_name = "Doe"
   last_name = "Smith"
   first_seen = 1609459200000000000
   last_updated = 1609459200000000000
   source_id = 1

   # Insert the new patient record into the dataset
   new_patient_id = sdk.insert_patient(patient_id=patient_id, mrn=mrn, gender=gender, dob=dob,
                                       first_name=first_name, middle_name=middle_name, last_name=last_name,
                                       first_seen=first_seen, last_updated=last_updated, source_id=source_id)

Next, we will iterate through the records in the MIT-BIH Arrhythmia Database and write the data to our dataset.

.. code-block:: python

   import numpy as np

   for record in get_records():
       time_data = np.arange(len(record.p_signal), dtype=np.int64) * (10 ** 9) // freq
       value_data = record.p_signal[:, 0]

       # Write the data to the dataset
       sdk.write_data_easy(measure_id=new_measure_id, device_id=new_device_id, time_data=time_data, value_data=value_data, freq=freq, freq_units="Hz", time_units="s")

Querying Data from the Dataset
-------------------------------

Now that we have inserted data into our dataset, let's query the data and perform some basic analysis.

First, let's retrieve information about the measures, devices, and patients in our dataset.

.. code-block:: python

   # Retrieve information about all measures in the dataset
   all_measures = sdk.get_all_measures()
   print(all_measures)

   # Retrieve information about all devices in the dataset
   all_devices = sdk.get_all_devices()
   print(all_devices)

   # Retrieve information about all patients in the dataset
   all_patients = sdk.get_all_patients()
   print(all_patients)

Next, let's retrieve the interval arrays for our measure and device.

.. code-block:: python

   interval_arr_device = sdk.get_interval_array(measure_id=new_measure_id, device_id=new_device_id)
   print(interval_arr_device)

Now, let's query the data for a specific time range and perform some basic analysis.

.. code-block:: python

   start_epoch_s = 0
   end_epoch_s = start_epoch_s + 3600  # 1 hour after start.
   start_epoch_nano = start_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
   end_epoch_nano = end_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds

   _, r_times, r_values = sdk.get_data(measure_id=new_measure_id, start_time_n=start_epoch_nano, end_time_n=end_epoch_nano, device_id=new_device_id)

   # Perform some basic analysis on the retrieved data, such as calculating the mean and standard deviation of the ECG signal.

.. code-block:: python

   mean_ecg = np.mean(r_values)
   std_ecg = np.std(r_values)

   print(f"Mean ECG value: {mean_ecg}")
   print(f"Standard deviation of ECG values: {std_ecg}")
