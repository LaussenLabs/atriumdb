################################################################################
Tutorials
################################################################################

***************************************
Standard Data Access
***************************************

In this tutorial, we will walk you through the process of creating a dataset, inserting new data, and querying that
data using the atriumdb library. In this example we will pull data from the MIT-BIH Arrhythmia Database and store it in our dataset.

Prerequisites
-------------

- Python 3.10 or higher
- atriumdb library
- wfdb library
- matplotlib library
- tqdm library

You can install the required libraries using pip:

.. code-block:: bash

   pip install atriumdb wfdb tqdm matplotlib

Creating a New Dataset
----------------------

First, let's create a new dataset using the atriumdb library. We will use the default SQLite metadata database for simplicity.
The :ref:`create_dataset <create_dataset_label>` method asks you to specify:

- `dataset_location`: The local directory where the binary files will be written.
- `database_type`: What type of supporting database technology to use (sqlite is the default, mariadb, mysql).
- `connection_params`: If using mariadb or mysql, connection parameters described below used to connect to the database.

.. code-block:: python

   from atriumdb import AtriumSDK

   sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset")

You can also create a dataset with a different metadata database, such as MariaDB or MySQL, by providing the
`database_type` and `connection_params` parameters. For example:

.. code-block:: python

   connection_params = {
       'host': "localhost",
       'user': "user",
       'password': "pass",
       'database': "new_dataset",
       'port': 3306
   }
   sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", connection_params=connection_params)


Inserting Data into the Dataset
--------------------------------

Now that we have created a new dataset, let's insert some data into it. Below, we read data
from the MIT-BIH Arrhythmia Database and store it in our dataset. In this example, we will create a separate device
for each record and handle multiple signals in a single record.

.. code-block:: python

    import wfdb
    from tqdm import tqdm
    import numpy as np

    # Get the list of record names from the MIT-BIH Arrhythmia Database
    pn_dir = 'mitdb'
    record_names = wfdb.get_record_list(pn_dir)

    # Loop through each record in the record_names list and read the record using the `rdrecord` function from the wfdb library
    for n in tqdm(record_names):
        # Pull record with digital values
        record = wfdb.rdrecord(n, pn_dir=pn_dir, return_res=64, physical=False)

        # For each record, create a new device in our dataset with the record name as the device tag
        # Check if a device with the given tag already exists using the `get_device_id` function
        # If it doesn't exist, create a new device using the `insert_device` function
        device_id = sdk.get_device_id(device_tag=record.record_name)
        if device_id is None:
            device_id = sdk.insert_device(device_tag=record.record_name)

        # Read The Record Annotations
        annotation = wfdb.rdann(n, 'atr', pn_dir="mitdb", summarize_labels=True, return_label_elements=['description'])
        label_time_idx_array = annotation.sample
        label_time_array = label_time_idx_array * (1 / record.fs)
        label_value_list = annotation.description

        # Define list of labels for the record
        labels = []

        # Create labels for each annotation
        for i in range(len(label_value_list)):
            start_time = label_time_array[i]
            end_time = start_time + (1 / record.fs)  # Assuming an annotation lasts for one sample
            label_name = label_value_list[i]
            label_measure_id = None  # No specific signal associated with this label.
            label_source = 'WFDB Arrhythmia Annotation'  # Where the label came from
            labels.append((label_name, label_source, device_id, label_measure_id, start_time, end_time))

        # Insert labels into the database
        sdk.insert_labels(labels=labels, time_units='s', source_type='device_id')

        # If there are multiple signals in one record, split them into separate dataset entries
        start_time_s = 0
        if record.n_sig > 1:
            for i in range(len(record.sig_name)):

                # Check if a measure with the given tag and frequency already exists in the dataset using the `get_measure_id` function
                # If it doesn't exist, create a new measure using the `insert_measure` function
                measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, unit=record.units[i], freq_units="nHz")
                if measure_id is None:
                    measure_id = sdk.insert_measure(measure_tag=record.sig_name[i], freq=freq_nano, unit=record.units[i], freq_units="nHz")

                # Calculate the digital to analog scale factors.
                gain = segment.adc_gain[i]
                baseline = segment.baseline[i]
                scale_m = 1 / gain
                scale_b = -baseline / gain

                # Write the data using the `write_segment` function
                sdk.write_segment(measure_id, device_id, record.d_signal.T[i], start_time_s, freq=record.fs,
                    scale_m=scale_m, scale_b=scale_b, time_units="s", freq_units="Hz")

        # If there is only one signal in the input file, insert it in the same way as for multiple signals
        else:
            # Check if a measure with the given tag and frequency already exists in the dataset using the `get_measure_id` function
            # If it doesn't exist, create a new measure using the `insert_measure` function
            measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, unit=record.units)
            if measure_id is None:
                measure_id = sdk.insert_measure(measure_tag=record.sig_name, freq=freq_nano, unit=record.units)

            # Calculate the digital to analog scale factors.
            gain = segment.adc_gain
            baseline = segment.baseline
            scale_m = 1 / gain
            scale_b = -baseline / gain

            # Write the data using the `write_data_easy` function
            sdk.write_segment(measure_id, device_id, record.d_signal, start_time_s, freq=record.fs, scale_m=scale_m, scale_b=scale_b,
                time_units="s", freq_units="Hz")

.. _methods_of_inserting_data:

Methods of Inserting Data
--------------------------

There are multiple ways to insert data into AtriumDB, depending on the format and use case.

The two primary methods are: inserting **segments** and inserting **time-value pairs**, both with the option of using
**buffered inserts** to batch small pieces of data together.

Understanding these formats helps to select the best approach for your use case.

Segments
^^^^^^^^^^

Segments are `a sequence of evenly-timed samples <https://en.wikipedia.org/wiki/Sampling_(signal_processing)/>`_ .
A segment includes a **start time**, a **sampling frequency**, and a sequence of **values**.
The timestamp of each value can be inferred based on the start time and the frequency.

Segments are often used for high-frequency waveforms or signals.

Segments can be inserted one at a time using `AtriumSDK.write_segment <contents.html#atriumdb.AtriumSDK.write_segment>`_
or in batches using `AtriumSDK.write_segments <contents.html#atriumdb.AtriumSDK.write_segments>`_.

Segments can also be batched piece by piece using :ref:`buffered_inserts`.

.. code-block:: python

    sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
    measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
    device_id = sdk.insert_device(device_tag="test_device")

    # Inserting a single segment
    segment_values = np.arange(100)  # Continuous values from 0 to 99
    start_time = 0.0  # Start time in seconds
    sdk.write_segment(measure_id, device_id, segment_values, start_time, freq=1.0, time_units="s", freq_units="Hz")

    # Inserting multiple segments at once
    segments = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
    start_times = [0.0, 10.0, 20.0]  # Start times in seconds for each segment
    sdk.write_segments(measure_id, device_id, segments, start_times, freq=1.0, time_units="s", freq_units="Hz")


Time-Value Pairs
^^^^^^^^^^^^^^^^^^

Time-value pairs allow you to insert irregularly sampled data, where each value has its own specific timestamp.
This format is common for low-frequency signals, such as metrics or aperiodic signals.

The method `AtriumSDK.write_time_value_pairs <contents.html#atriumdb.AtriumSDK.write_time_value_pairs>`_
can be used for inserting time-value pairs, with arrays of values and corresponding timestamps passed as arguments.

.. code-block:: python

    sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
    measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
    device_id = sdk.insert_device(device_tag="test_device")

    # Inserting time-value pairs
    times = np.array([0.0, 2.0, 4.5])  # Time values in seconds
    values = np.array([100, 200, 300])  # Corresponding values
    sdk.write_time_value_pairs(measure_id, device_id, times, values, time_units="s")

.. _buffered_inserts:

Buffered Inserts
^^^^^^^^^^^^^^^^^^^^

Buffered inserts allow for efficient batch writing of data into the database.
When using the buffer, data is accumulated until a threshold is met (e.g., the number of values exceeds a specified maximum),
at which point the buffer is automatically flushed. The buffer can also be flushed manually and automatically upon exiting the buffer's context.
This method is optimal for live ingesting segments as they come from a device or back loading an archive of many small segments.

You can buffer both **segments** and **time-value pairs** using the `AtriumSDK.write_buffer <contents.html#atriumdb.AtriumSDK.write_buffer>`_ method.
The buffer organized data by their measure-device pair, and data is automatically written once the buffer fills or the context is closed.

.. code-block:: python

    sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
    measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
    device_id = sdk.insert_device(device_tag="test_device")

    # Using write_buffer for batched writes
    reasonable_num_values_per_value = 100 * sdk.block.block_size  # 100 blocks
    with sdk.write_buffer(max_values_per_measure_device=reasonable_num_values_per_value,
                          max_total_values_buffered=10 * reasonable_num_values_per_value) as buffer:
        # Write multiple small segments to buffer
        for record in record_segments:
            sdk.write_segment(measure_id, device_id, record.d_signal, start_time_s, freq=record.fs,
                              scale_m=scale_m, scale_b=scale_b, time_units="s", freq_units="Hz")

        buffer.flush_all()
        # Buffer auto-flushes when the context is exited

Surveying Data in the Dataset
-----------------------------

In this section, we will discuss how to survey the data in our dataset, including retrieving information about all
measures and devices, and obtaining the availability of specified measures and sources.

Retrieving All Measures
^^^^^^^^^^^^^^^^^^^^^^^

To retrieve information about all measures in the dataset, you can use the :ref:`get_all_measures <get_all_measures_label>` method.
This method queries the linked relational database and returns a dictionary containing detailed information about each measure stored in the dataset.

The information includes:

- `id`: The unique identifier of the measure in the dataset.
- `tag`: A short, human-readable identifier for the measure.
- `name`: A more descriptive name for the measure (can be None if not defined).
- `freq_nhz`: The sample frequency of the measure in nanohertz (1 Hz = 10^9 nHz).
- `code`: A code (usually CF_CODE10) representing the measure (can be None if not defined).
- `unit`: The unit of the measure (e.g., 'BPM' for beats per minute).
- `unit_label`: A human-readable label for the unit (can be None if not defined).
- `unit_code`: A code (usually CF_CODE10) representing the unit (can be None if not defined).
- `source_id`: The identifier of the data source (e.g., device or patient) associated with the measure.

Here's an example of how to use the :ref:`get_all_measures <get_all_measures_label>` method:

.. code-block:: python

   # Instantiate the AtriumSDK object with the dataset location
   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Retrieve information about all measures in the dataset
   all_measures = sdk.get_all_measures()

   # Print the retrieved information
   print(all_measures)

Example output:

.. code-block:: python

   {
       1: {
           'id': 1,
           'tag': 'MLII',
           'name': None,
           'freq_nhz': 360000000000,
           'code': None,
           'unit': 'mV',
           'unit_label': None,
           'unit_code': None,
           'source_id': 1
       },
       2: {
           'id': 2,
           'tag': 'V5',
           'name': None,
           'freq_nhz': 360000000000,
           'code': None,
           'unit': 'mV',
           'unit_label': None,
           'unit_code': None,
           'source_id': 1
       },
   }

In this example, the dataset contains two measures: ECG Lead MLII and ECG Lead V5,
both with a sample frequency of 360000000000 nanohertz (360 Hz) and units in millivolts (mV).

Retrieving All Devices
^^^^^^^^^^^^^^^^^^^^^^

To retrieve information about all devices in the dataset, you can use the :ref:`get_all_devices <get_all_devices_label>` method.
This method returns a dictionary containing information about each device in the dataset.

The information includes:

- `id`: The unique identifier of the device in the dataset.
- `tag`: A short, human-readable identifier for the device.
- `name`: A more descriptive name for the device (can be None if not defined).
- `manufacturer`: The manufacturer of the device (can be None if not defined).
- `model`: The model of the device (can be None if not defined).
- `type`: The type of the device (e.g., 'static', 'dynamic', 'monitor').
- `bed_id`: The identifier of the bed associated with the device (can be None if not defined).
- `source_id`: The identifier of the data source (e.g., device or patient) associated with the device.

Here's an example of how to use the :ref:`get_all_devices <get_all_devices_label>` method:

.. code-block:: python

   all_devices = sdk.get_all_devices()
   print(all_devices)

Example output:

.. code-block:: python

   {
       1: {
           'id': 1,
           'tag': '100',
           'name': None,
           'manufacturer': None,
           'model': None,
           'type': 'static',
           'bed_id': None,
           'source_id': 1
       },
       2: {
           'id': 2,
           'tag': '101',
           'name': None,
           'manufacturer': None,
           'model': None,
           'type': 'static',
           'bed_id': None,
           'source_id': 1
       },
       # ...
   }

In this example, the :ref:`get_all_devices <get_all_devices_label>` method returns a dictionary where the keys are the device ids and the values are
dictionaries containing the device properties. You can see that the output includes information about the
device's tag, name, manufacturer, model, type, bed_id, and source_id.

By examining the output, you can gain insights into the devices present in your dataset and their characteristics.
For example, you might notice that some devices have missing information (e.g., name, manufacturer, model),
which you could then decide to update or investigate further. Additionally, you can use the device ids to query your
dataset based on specific devices.

Getting Data Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^
To obtain the availability of a specified measure (signal) and a specified source (device id or patient id),
you can use the :ref:`get_interval_array <get_interval_array_label>` method. This method provides information about the available data for a specific measure
and source by returning a 2D array representing the data availability.

Each row of the 2D array output represents a continuous interval of available data, with the first and second columns
representing the start epoch and end epoch of that interval, respectively.
This information can be useful when you want to analyze or visualize data within specific time periods or when you need to identify gaps in the data.

Here's an example of how to use the :ref:`get_interval_array <get_interval_array_label>` method:

.. code-block:: python

   # Define the measure_id and device_id for which you want to get data availability
   measure_id = 1
   device_id = 1

   # Call the get_interval_array method
   interval_arr = sdk.get_interval_array(measure_id=measure_id, device_id=device_id)

   # Print the resulting 2D array
   print(interval_arr)

Example output:

.. code-block:: python

   [[            0 1805555050000]]

In this example, the output shows that there is a single continuous interval of available data for the specified measure and device,
starting at epoch 0 and ending at epoch 1805555050000. This is because there are no gaps in the source mit-bih data.

These methods allow you to survey the data in your dataset and obtain information about the measures, devices, and data availability.
By understanding the data availability, you can make informed decisions about how to process, analyze, or visualize the data in your dataset.

Querying Data from the Dataset
-------------------------------

Now that we have inserted and surveyed the data into our dataset, let's query the data and verify that the data has been correctly inserted.
We will iterate through the records in the MIT-BIH Arrhythmia Database and compare the data in our dataset to the original data.

.. code-block:: python

   # Iterate through the record names in the MIT-BIH Arrhythmia Database
   for n in tqdm(record_names):

       # Read the record from the MIT-BIH Arrhythmia Database
       record = wfdb.rdrecord(n, pn_dir="mitdb")
       # Calculate the sample frequency in nanohertz
       freq_nano = record.fs * 1_000_000_000

       # Get the device ID for the current record
       device_id = sdk.get_device_id(device_tag=record.record_name)

       # If there are multiple signals in the record, check both
       if record.n_sig > 1:
           for i in range(len(record.sig_name)):
               # Get the measure ID for the current signal
               measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])

               # Query the data from the dataset
               _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)

               # Check that the signal from MIT-BIH and AtriumDB are equal
               assert np.allclose(record.p_signal.T[i], read_values)

       # If there is only one signal in the record
       else:
           # Get the measure ID for the signal
           measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)

           # Query the data from the dataset
           _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)

           # Check that the signal from MIT-BIH and AtriumDB are equal
           assert np.allclose(record.p_signal.T[i], read_values)


Retrieving Labels from the Dataset
------------------------------------------

We can also retrieve the annotations inserted as atriumdb labels earlier in the tutorial, first by recalling the different
label names inserted into the dataset:

.. code-block:: python

    label_name_dict = sdk.get_all_label_names()
    label_names = [label_info['name'] for label_id, label_info in label_name_dict.items()]

And then by calling `AtriumSDK.get_labels` to retrieve the label information:

.. code-block:: python

    for record_name in tqdm(record_names):
       # Read the record from the MIT-BIH Arrhythmia Database
       label_data = sdk.get_labels(name_list=label_names, device_list=[record_name])

Visualizing the Dataset
-------------------------------

Finally, let's retrieve data from our dataset and plot the first 1000 points of the first record's data.
We will use the `matplotlib` library to create a simple line plot of the data.

.. code-block:: python

    import matplotlib.pyplot as plt

    # Define the measure_id and device_id we want to retrieve data for
    measure_id = 1
    device_id = 1

    # Get the measure information for the specified measure_id
    measure_info = sdk.get_measure_info(measure_id=measure_id)
    device_info = sdk.get_device_info(device_id=device_id)

    # Extract the frequency in nanohertz from the measure information
    freq_nhz = measure_info['freq_nhz']

    # Calculate the period in nanoseconds by dividing 10^18 by the frequency in nanohertz
    period_nhz = int((10 ** 18) // freq_nhz)

    # Define the start and end time for the data we want to retrieve
    # We want to retrieve the first 1000 points, so we set the end time to 1001 times the period
    start_time_n, end_time_n = 0, 1001 * period_nhz  # [start, end)

    # Retrieve the data for the specified measure_id, device_id, start_time_n, and end_time_n
    _, times, values = sdk.get_data(measure_id=measure_id, device_id=device_id, start_time_n=start_time_n,
                                    end_time_n=end_time_n)

    # Plot the first 1000 points of the first patient's data using matplotlib
    plt.plot(times / (10 ** 9), values)  # convert x-axis units to seconds.
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Signal Value")
    plt.title(f"First 1000 Points of Measure {measure_info['tag']} and Device {device_info['tag']}")
    plt.show()

.. image:: mit_bih_1000_samples.png
   :alt: ECG plot
   :align: center


************************************************
Reading Dataset With Iterators
************************************************

Working with large datasets often requires efficient access to smaller windows of data, particularly for tasks such
as data visualization, pre-processing, or model training. The AtriumSDK provides a convenient method, `get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_,
to handle these cases effectively.

Creating a Dataset Definition
-----------------------------

The `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object specifies the measures, patients and/or devices, and the time intervals we are interested in querying.
This definition can be provided in two different ways: by reading from a YAML file or by creating the object in your Python script.

**Option 1: Using a YAML file**

Suppose you have the following in your `definition.yaml  <dataset.html#definition-file-format>`_ file:

.. code-block:: yaml

    device_ids:
      1: all
      2: all

    measures:
      - MLII
      - tag: V1
        freq_hz: 360.0
        units: 'mV'

You can load this into a `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object as follows:

.. code-block:: python

    from atriumdb import DatasetDefinition

    definition = DatasetDefinition(filename="definition.yaml")


**Option 2: Creating an object via Python script**

Alternatively, you can define your dataset programmatically:

.. code-block:: python

    from atriumdb import DatasetDefinition

    measures = ['MLII',
                {"tag": "V1", "freq_hz": 360.0, "units": "mV"},]
    device_ids = {
        1: 'all',
        2: 'all',
    }

    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

If you wanted to create a dataset of all patients born after a certain date, you could setup your patient_ids dictionary like:

.. code-block:: python

    min_dob = 1572739200000000000  # Nanosecond epoch
    patient_ids = {patient_id: "all" for patient_id, patient_info in
        sdk.get_all_patients().items() if patient_info['dob'] and patient_info['dob'] > min_dob}

    definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)


**Generating a DatasetDefinition for WFDB Example**

.. code-block:: python

    measures = [{"tag": measure_info['tag'],
                 "freq_nhz": measure_info['freq_nhz'],  # Can specify freq_nhz or freq_hz
                 "units": measure_info['unit']}
                for measure_info in sdk.get_all_measures().values()]
    device_ids = {device_id: 'all' for device_id in sdk.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

Iterating Over Windows
----------------------

Now that we've setup the `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object, we can use it to iterate over our dataset.

.. code-block:: python

    window_size = 60
    slide_size = 30

    # Obtain the iterator
    iterator = sdk.get_iterator(definition, window_size, slide_size, time_units="s")

    # Now you can iterate over the data windows
    for window_i, window in enumerate(iterator):
        print(f"Window: {window_i}")
        print(f"Start Time: {window.start_time}")
        print(f"Device ID: {window.device_id}")
        print(f"Patient ID: {window.patient_id}")

        # Use window.signals to view available signals in their original form
        for (measure_tag, measure_freq_hz, measure_units), signal_dict in window.signals.items():
            print(f"Measure: {measure_tag}, Frequency: {measure_freq_hz} Hz, Units: {measure_units}")
            print(f"Times: {signal_dict['times']}")
            print(f"Values: {signal_dict['values']}")
            print(f"Expected Count: {signal_dict['expected_count']}")
            print(f"Actual Count: {signal_dict['actual_count']}")


***************************************
Full Tutorial Script
***************************************

You can view or download the full Python script used in this tutorial here :download:`tutorial_script.py <scripts/tutorial_script.py>`.