Querying Data from a Dataset
######################################

Measures
~~~~~~~~~~~~~~~~

The following methods allow you to retrieve information about the measures (signal types) in your dataset:

- :ref:`get_all_measures <get_all_measures_label>`: Retrieve information about all measures in the dataset.
- :ref:`get_measure_info <get_measure_info_label>`: Retrieve information about a specific measure by its ID.
- :ref:`get_measure_id <get_measure_id_label>`: Retrieve the ID of a measure by its tag.

Examples:

.. code-block:: python

   from atriumdb import AtriumSDK

   # Initialize the AtriumSDK object
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Retrieve information about all measures in the dataset
   all_measures = sdk.get_all_measures()
   print(all_measures)  # {1: {'id': 1, 'tag': 'Heart Rate', ...}, 2: {...}, ...}

   # Retrieve information about a specific measure by its ID
   measure_id = 1
   measure_info = sdk.get_measure_info(measure_id)
   print(measure_info)  # {'id': 1, 'tag': 'Heart Rate', 'name': 'Heart rate in beats per minute', ...}

   # Retrieve the ID of a measure by its tag
   measure_tag = "Heart Rate"
   measure_id = sdk.get_measure_id(measure_tag)
   print(measure_id)  # 1

Devices
~~~~~~~~~~~~~~~~

The following methods allow you to retrieve information about the devices (data sources) in your dataset:

- :ref:`get_all_devices <get_all_devices_label>`: Retrieve information about all devices in the dataset.
- :ref:`get_device_info <get_device_info_label>`: Retrieve information about a specific device by its ID.
- :ref:`get_device_id <get_device_id_label>`: Retrieve the ID of a device by its tag.

Examples:

.. code-block:: python

   from atriumdb import AtriumSDK

   # Initialize the AtriumSDK object
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Retrieve information about all devices in the dataset
   all_devices = sdk.get_all_devices()
   print(all_devices)  # {1: {'id': 1, 'tag': 'Monitor A1', ...}, 2: {...}, ...}

   # Retrieve information about a specific device by its ID
   device_id = 1
   device_info = sdk.get_device_info(device_id)
   print(device_info)  # {'id': 1, 'tag': 'Device A1', 'name': 'Philips Device A1 in Room 1A', ...}

   # Retrieve the ID of a device by its tag
   device_tag = "Monitor A1"
   device_id = sdk.get_device_id(device_tag)
   print(device_id)  # 1

Patients
~~~~~~~~~~~~~~~~

The :ref:`get_all_patients <get_all_patients_label>` method allows you to retrieve information about all the patients in your dataset.
This method returns a dictionary containing information about each patient,
including their id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, and source_id.

Example:

.. code-block:: python

   from atriumdb import AtriumSDK

   # Initialize the AtriumSDK object
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Retrieve information about all patients in the dataset
   all_patients = sdk.get_all_patients()
   print(all_patients)  # {1: {'id': 1, 'mrn': 123456, ...}, 2: {...}, ...}

Interval Arrays
************************

Because our data is only retrievable when queried by specific time intervals, it is helpful to gain a larger picture view of what data is available in your dataset and when. The :ref:`get_interval_array <get_interval_array_label>` method returns a 2D array representing the availability of a specified measure (signal) and a specified source (device id or patient id). Each row of the 2D array output represents a continuous interval of available data, while the first and second columns represent the start epoch and end epoch of that interval, respectively.

Examples:

.. code-block:: python

   from atriumdb import AtriumSDK

   # Initialize the AtriumSDK object
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Retrieve intervals by device
   measure_id = 21
   device_id = 25
   interval_arr_device = sdk.get_interval_array(measure_id=measure_id, device_id=device_id)
   print(interval_arr_device)  # [[1669668855000000000, 1669668856000000000], [1669668857000000000, 1669668858000000000], ...]

   # Retrieve intervals by patient
   patient_id = 2
   interval_arr_patient = sdk.get_interval_array(measure_id=measure_id, patient_id=patient_id)
   print(interval_arr_patient)  # [[1669668855000000000, 1669668856000000000], [1669668857000000000, 1669668858000000000], ...]

   # Retrieve intervals within a specific time range
   start_epoch_nano = 1669668855000000000
   end_epoch_nano = start_epoch_nano + 3600 * (10 ** 9)
   interval_arr_time = sdk.get_interval_array(measure_id=measure_id, device_id=device_id, start=start_epoch_nano, end=end_epoch_nano)
   print(interval_arr_time)  # [[1669668855000000000, 1669668856000000000], [1669668857000000000, 1669668858000000000], ...]

Getting Data
************************

The :ref:`get_data <get_data_label>` method is used to query data from the dataset, indexed by signal type (measure_id),
time (start_time_n and end_time_n), and data source (device_id and patient_id).
This method returns a tuple containing a list of block header Python objects, a numpy 1D array representing
the time data (usually an array of timestamps), and a numpy 1D array representing the value data.

Examples:

.. code-block:: python

   from atriumdb import AtriumSDK

   # Initialize the AtriumSDK object
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Query data by device
   measure_id = 1
   device_id = 4
   start_epoch_s = 1669668855
   end_epoch_s = start_epoch_s + 3600  # 1 hour after start.
   start_epoch_nano = start_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
   end_epoch_nano = end_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
   _, r_times, r_values = sdk.get_data(measure_id=measure_id, start_time_n=start_epoch_nano, end_time_n=end_epoch_nano, device_id=device_id)
   print(r_times)
   print(r_values)

   # Expected output:
   # array([1669668855000000000, 1669668856000000000, 1669668857000000000, ...,
   #        1669672452000000000, 1669672453000000000, 1669672454000000000], dtype=int64)
   # array([ 0.32731968,  0.79003189,  0.99659552, ..., -0.59080797,
   #        -0.93542358, -0.97675089])

   # Query data by patient
   patient_id = 2
   _, r_times_patient, r_values_patient = sdk.get_data(measure_id=measure_id, start_time_n=start_epoch_nano, end_time_n=end_epoch_nano, patient_id=patient_id)
   print(r_times_patient)
   print(r_values_patient)

   # Expected output:
   # array([1669668855000000000, 1669668856000000000, 1669668857000000000, ...,
   #        1669672452000000000, 1669672453000000000, 1669672454000000000], dtype=int64)
   # array([ 0.12345678,  0.23456789,  0.34567890, ..., -0.45678901,
   #        -0.56789012, -0.67890123])