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

We will start by pulling a record from the MITDB database using the `wfdb` library.

.. code-block:: python

    import numpy as np
    import wfdb  # pip install wfdb

    # Fetch record 100 from MITDB
    record_name, pn_dir = "100", "mitdb"

    # Read the record (digital format)
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=False, physical=False)
    segments = record.segments if isinstance(record, wfdb.MultiRecord) else [record]

`insert_device`, `insert_measure` and `write_data_easy`: Defining a Device and Inserting Signals into AtriumDB
################################################################################################################

Each signal from the WFDB record will be stored in AtriumDB as a measure. We will first define a device, and then insert the signals.

.. code-block:: python

    # Define a new device in the database
    device_tag = "MITDB_record_100"
    device_id = sdk.insert_device(device_tag=device_tag)

    # Iterate over the WFDB segments to extract and store signal data
    end_frame = 0
    for segment in segments:
        start_frame = end_frame
        end_frame += segment.sig_len

        if segment.sig_len == 0:
            continue

        for i, measure_tag in enumerate(segment.sig_name):
            freq_hz = segment.fs * segment.samps_per_frame[i]
            start_time_s = start_frame / segment.fs
            gain = segment.adc_gain[i]
            baseline = segment.baseline[i]
            digital_signal = segment.e_d_signal[i]

            # Create a timestamp array
            time_data_s = np.arange(digital_signal.size) / freq_hz + start_time_s

            # Insert the signal (measure) into AtriumDB
            measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, freq_units="Hz")

            # Write the signal data to AtriumDB
            scale_m = 1 / gain
            scale_b = -baseline / gain
            sdk.write_data_easy(measure_id, device_id, time_data_s, digital_signal, freq_hz, scale_m=scale_m, scale_b=scale_b, time_units="s", freq_units="Hz")

`get_data`: Checking the written data against the source of truth
############################################################################

Once the digital signal is stored in AtriumDB, we can repull the record with its physical values to verify the data. Here’s how to do that:

.. code-block:: python

    # Repull the record in analog (physical) format
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=64, smooth_frames=False, m2s=True, physical=True)

    for i, measure_tag in enumerate(record.sig_name):
        analog_signal = record.e_p_signal[i]
        freq_hz = record.fs * record.samps_per_frame[i]

        # Retrieve the data from AtriumDB
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


Full Python Script
###################

You can view or download the full Python script used in this tutorial from the following link:

Download the full script :download:`quickstart_script.py <scripts/quickstart_script.py>`


Using the CLI for authentication and remote access
##################################################

To use the CLI for authentication and remote access, you will need to install the `atriumdb` package with the `cli` and `remote` optional dependency.

.. code-block:: bash

    pip install atriumdb[cli,remote]

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

The `atriumdb` CLI also provides commands for working with local datasets. You can use the CLI to list and filter measures, devices, and patients in your local dataset.

First, ensure you have the `atriumdb` package with the `cli` optional dependency installed:

.. code-block:: bash

    pip install atriumdb[cli]

To list measures, use the `measure ls` command:

.. code-block:: bash

    atriumdb measure ls

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