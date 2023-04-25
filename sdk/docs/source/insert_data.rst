Inserting data into a dataset
######################################

Measures, Devices, and Patients
********************************

Measures, devices, and patients are essential components of the dataset. Measures represent the signal types,
devices represent the electronic recording devices, and patients represent the individuals associated with the data.
Defining these components allows us to query simultaneous information from multiple data sources and understand
our dataset as it evolves over time.

Insert Measure
~~~~~~~~~~~~~~~~

The :ref:`insert_measure <insert_measure_label>` method is used to define a new signal type to be stored in the dataset, as well as defining metadata related to the signal. The method requires the sample frequency of the signal and allows for optional parameters such as measure_tag, measure_name, and units.

Example:

.. code-block:: python

   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Define a new signal.
   freq = 500
   freq_units = "Hz"
   measure_tag = "ECG Lead II - 500 Hz"
   measure_name = "Electrocardiogram Lead II Configuration 500 Hertz"
   units = "mV"

   # Insert the new signal into the dataset.
   new_measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units, measure_name=measure_name)

Insert Device
~~~~~~~~~~~~~~

The :ref:`insert_device <insert_device_label>` method is used to define a new data source to be stored in the dataset, as well as defining metadata related to the source. The method requires a unique device identifier and allows for optional parameters such as device_tag and device_name.

Example:

.. code-block:: python

   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Define a new data source.
   device_tag = "Monitor A3"
   device_name = "Philips Monitor A3 in Room 2B"

   # Insert the new data source into the dataset.
   new_device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)

Insert Patient
~~~~~~~~~~~~~~~

The :ref:`insert_patient <insert_patient_label>` method is used to insert a new patient record into the database with the provided patient details. All patient details are optional, but it is recommended to provide as much information as possible to ensure accurate patient identification and to avoid duplicate records.

Example:

.. code-block:: python

   sdk = AtriumSDK(dataset_location="./example_dataset")

   # Define a new patient record.
   patient_id = 123
   mrn = "123456"
   gender = "M"
   dob = 946684800000000000
   first_name = "John"
   middle_name = "Doe"
   last_name = "Smith"
   first_seen = 1609459200000000000
   last_updated = 1609459200000000000
   source_id = 1

   # Insert the new patient record into the dataset.
   new_patient_id = sdk.insert_patient(patient_id=patient_id, mrn=mrn, gender=gender, dob=dob,
                                       first_name=first_name, middle_name=middle_name, last_name=last_name,
                                       first_seen=first_seen, last_updated=last_updated, source_id=source_id)

Writing Data Example
************************

To write data to the dataset, you can use either the :ref:`write_data <write_data_label>` method for advanced usage or the :ref:`write_data_easy <write_data_easy_label>` method for a simplified approach. Both methods require measure_id, device_id, time_data, and value_data as parameters. Additional parameters are available for customization and optimization, such as raw_time_type, raw_value_type, encoded_time_type, encoded_value_type, scale_m, and scale_b.

Example using :ref:`write_data_easy <write_data_easy_label>`:

.. code-block:: python

    import numpy as np
    from atriumdb import AtriumSDK

    # Initialize the AtriumSDK object
    sdk = AtriumSDK(dataset_location="./example_dataset")

    # Define the measure and device IDs
    new_measure_id = 21
    new_device_id = 21

    # Create some time data (in seconds)
    freq_hz = 1
    time_data = np.arange(1234567890, 1234567890 + 3600, dtype=np.int64)

    # Create some value data of equal dimension
    value_data = np.sin(time_data)

    # Write the data to the dataset
    sdk.write_data_easy(measure_id=new_measure_id, device_id=new_device_id, time_data=time_data, value_data=value_data, freq=freq_hz, freq_units="Hz", time_units="s")

This example demonstrates how to use the :ref:`write_data_easy <write_data_easy_label>` method to write data to the dataset. The time and value data are generated using NumPy, and the measure and device IDs are specified. The data is then written to the dataset using the specified frequency and time units.

Mapping Devices to Patients
*****************************

The :ref:`insert_device_patient_data <insert_device_patient_data_label>` method is used to associate devices with patients in the dataset's database. This method takes a list of tuples as input, where each tuple contains four integer values in the following order:

1. device_id: The ID of the device associated with the patient.
2. patient_id: The ID of the patient associated with the device.
3. start_time: The start time of the association between the device and the patient, represented in UNIX nano timestamp format.
4. end_time: The end time of the association between the device and the patient, also represented in UNIX nano timestamp format.

The `start_time` and `end_time` values define the time range during which the device is associated with the patient.

Example:

.. code-block:: python

    from atriumdb import AtriumSDK

    # Initialize the AtriumSDK object.
    sdk = AtriumSDK(dataset_location="./example_dataset")

    # Define the device-patient mappings.
    device_patient_data = [
        (1, 2, 1647084000_000_000_000, 1647094800_000_000_000),
        (1, 3, 1647094800_000_000_000, 1647104800_000_000_000)
    ]

    # Insert the device-patient mappings into the dataset's database.
    sdk.insert_device_patient_data(device_patient_data)

In this example, we have two device-patient mappings.
The first tuple maps device 1 to patient 2, with a start time of 1647084000_000_000_000 and an end time of 1647094800_000_000_000.
The second tuple maps the same device 1 to patient 3, with different start and end times.
The `insert_device_patient_data` method is then called to insert these mappings into the dataset's database.
