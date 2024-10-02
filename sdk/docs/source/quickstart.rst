Quick Start
-------------

This quick start guide will walk you through the process of creating a new dataset, pulling example data from WFDB, defining signals and sources, and reading and writing data in AtriumDB.

Creating a New Dataset
#######################

To create a new dataset, you can use the `create_dataset` method. This method allows you to specify the type of metadata database to use and where the data will be stored.

.. code-block:: python

    import os
    import shutil
    from atriumdb import AtriumSDK

    # Define dataset location
    dataset_location = "./new_dataset"

    # Reset the local database if it exists
    if os.path.exists(dataset_location):
        shutil.rmtree(dataset_location)

    # Create a new dataset
    sdk = AtriumSDK.create_dataset(dataset_location=dataset_location)

Connecting to an Existing Dataset
#######################################

To connect to an already created dataset, you will need to specify a local path where the dataset is stored if it's a SQLite database.
If it's a MariaDB dataset, you will also have to specify the connection parameters.

.. code-block:: python

    # Import AtriumSDK python object
    from atriumdb import AtriumSDK

    # Define a directory path where the dataset is stored (always needed)
    dataset_location = "./example_dataset"

    # Create AtriumSDK python object (SQLite)
    sdk = AtriumSDK(dataset_location=dataset_location)

    # OR Connect to a dataset supported by MariaDB
    connection_params = {
        'host': "localhost",
        'user': "user",
        'password': "pass",
        'database': "new_dataset",
        'port': 3306
    }
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type="mysql", connection_params=connection_params)

    # Connect to a remote dataset using the API
    api_url = "http://example.com/v1"
    token = "4e78a93749ead7893"
    sdk = AtriumSDK(api_url=api_url, token=token, metadata_connection_type="api")


Pulling Example Data from WFDB
#######################################

We will start by pulling a record from the MITDB database using the `wfdb` library. AtriumDB indexes data by
`time epochs <https://www.epochconverter.com/>`_ so we'll also manufacture some time information.

.. code-block:: python

    import numpy as np
    import wfdb  # pip install wfdb

    # Fetch record 100 from MITDB
    record_name, pn_dir = "100", "mitdb"

    # Read the record (digital format)
    # AtriumDB performs significantly better when signals are written as digital integers with associated scale factors.
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=False, physical=False)
    segments = record.segments if isinstance(record, wfdb.MultiRecord) else [record]

Inserting Signals
######################################################################################################################

Each signal from the WFDB record will be stored in AtriumDB as a measure. We will first define a device, and then insert the signals.

Time information from WFDB and many medical monitors are provided as a start time, sample frequency and a sequence of
`sequential signal values <https://en.wikipedia.org/wiki/Sampling_(signal_processing)/>`_  seperated by a constant
sample period defined by the sample frequency.

In that case we use the `AtriumSDK.write_segment  <contents.html#atriumdb.AtriumSDK.write_segment>`_ method.
For inserting data of alternate time formats (for example, time-value pairs), see :ref:`methods_of_inserting_data`.

.. code-block:: python

    # Define a new device in the database. If the device already exists, the id will simply be returned.
    device_tag = "MITDB_record_100"
    device_id = sdk.insert_device(device_tag=device_tag)

    # Iterate over the WFDB segments to extract and store signal data
    end_frame = 0
    for segment in segments:
        start_frame = end_frame
        end_frame += segment.sig_len

        if segment.sig_len == 0:
            continue

        for i, signal_name in enumerate(segment.sig_name):
            freq_hz = segment.fs * segment.samps_per_frame[i]
            start_time_s = start_frame / segment.fs
            gain = segment.adc_gain[i]
            baseline = segment.baseline[i]
            digital_signal = segment.e_d_signal[i]

            # Define a new signal type (measure) in AtriumDB. If the signal already exists, the id will be returned
            # without defining anything new. `freq_units` must be specified!
            measure_id = sdk.insert_measure(measure_tag=signal_name, freq=freq_hz, freq_units="Hz")

            # Scale factors such that: Analog_Signal = scale_m * Digital_Signal + scale_b
            scale_m = 1 / gain
            scale_b = -baseline / gain

            # Write the signal data to AtriumDB
            sdk.write_segment(measure_id, device_id, digital_signal, start_time_s, freq=freq_hz, scale_m=scale_m, scale_b=scale_b)

Querying Data
############################################################################

Once the digital signal + scale factors are stored in AtriumDB, we can repull the record with its physical/analog values to verify the data.

.. code-block:: python

    # Repull the record in analog (physical) format
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=True, physical=True)

    for i, measure_tag in enumerate(record.sig_name):
        analog_signal = record.e_p_signal[i]
        freq_hz = record.fs * record.samps_per_frame[i]

        # Retrieve the data from AtriumDB. `time_units` must be specified as the default is "ns" nanoseconds.
        measure_id = sdk.get_measure_id(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")
        _, read_time_data, read_value_data = sdk.get_data(
            measure_id=measure_id,
            start_time_n=0,
            end_time_n=end_frame / record.fs,
            device_id=device_id,
            time_units="s",
        )

        # Verify the data matches the original analog signal
        assert np.allclose(read_value_data, analog_signal)


Full Quick Start Script
###########################

You can view or download the full Python script used in this tutorial here :download:`quickstart_script.py <scripts/quickstart_script.py>`.


Using the CLI for authentication and remote access
##################################################

To use the CLI for authentication and remote access, you will need to install the `atriumdb` package with the `cli` and `remote` optional dependency.

.. code-block:: bash

    pip install atriumdb[cli,remote]
    # or pip install atriumdb[all]

You can then use the `atriumdb` CLI to set the endpoint URL and log in to the remote API.

.. code-block:: bash

    atriumdb login --endpoint-url "https://example.com/v1"

This command, after authenticating your API connection, will save your URL, token, auth expiration time, and connection mode in the `.env`:

.. code-block:: ini

    ATRIUMDB_ENDPOINT_URL=https://example.com/v1
    ATRIUMDB_API_TOKEN='aBcD012345eFgHI'
    ATRIUMDB_AUTH_EXPIRATION_TIME=1234567890.1234567
    ATRIUMDB_DATABASE_TYPE='api'

Once these variables have been set after running `login`, you can refresh the token using:

.. code-block:: bash

    atriumdb refresh-token

Now, you can access the remote dataset using the AtriumSDK object, as shown in the "Connecting to an Existing Dataset" section.


Using the CLI for Local Operations
##################################

The `atriumdb` CLI also provides commands for working with local datasets. You can use the CLI to list data available and export datasets.

You will only need the `cli` optional dependency installed:

.. code-block:: bash

    pip install atriumdb[cli]

Assuming you have an atriumdb dataset in the current working directory:

To list measures, use the `measure ls` command:

.. code-block:: bash

    atriumdb --dataset-location . measure ls

Assuming you have the ATRIUMDB_DATASET_LOCATION environment variable set to the `dataset_location` of
your atriumdb dataset:

To filter measures by a specific tag or frequency, use the `--tag-match` or `--freq` options:

.. code-block:: bash

    atriumdb measure ls --tag-match "ECG" --freq 250

To list devices, use the `device ls` command:

.. code-block:: bash

    atriumdb device ls

To filter devices by a specific tag or manufacturer, use the `--tag-match` or `--manufacturer-match` options:

.. code-block:: bash

    atriumdb device ls --tag-match "monitor" --manufacturer-match "Philips"

To list patients, use the `patient ls` command:

.. code-block:: bash

    atriumdb patient ls

To filter patients by gender or age range, use the `--gender` or `--age-years-min` and `--age-years-max` options:

.. code-block:: bash

    atriumdb patient ls --gender "F" --age-years-min 20 --age-years-max 40