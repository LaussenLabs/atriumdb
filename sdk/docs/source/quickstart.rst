Quick Start
-------------

This quick start guide will walk you through the process of creating a new dataset, pulling example data from WFDB, defining signals and sources, and reading and writing data in AtriumDB.

Creating a new dataset
#######################

To create a new dataset, you can use the `create_dataset` method. This method allows you to specify the type of metadata database to use and where the data will be stored.

.. code-block:: python

    from atriumdb import AtriumSDK

    # Create a new local dataset using SQLite
    sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset")

    # OR create a new local dataset using MariaDB
    connection_params = {
        'host': "localhost",
        'user': "user",
        'password': "pass",
        'database': "new_dataset",
        'port': 3306
    }
    sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", connection_params=connection_params)

Connecting to an Existing Dataset
#######################################

To connect to an already created dataset, you will need to specify a local path where the dataset is stored if it's a sqlite database.
If it's a MariaDB dataset you will also have to specify the connection parameters.

.. code-block:: python

    # Import AtriumSDK python object
    from atriumdb import AtriumSDK

    # Define a directory path where the dataset is stored (always needed)
    dataset_location = "./example_dataset"

    # Create AtriumSDK python object (sqlite)
    sdk = AtriumSDK(dataset_location=dataset_location)

    # OR Connect to a dataset supported by mariadb
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


Pull Some Example Data from WFDB
#####################################

To pull some example data from WFDB, you will need to install the `wfdb` library.

.. code-block:: bash

    pip install wfdb

Then, you can use the `wfdb` library to read in a record from the MITDB database.

.. code-block:: python

    import numpy as np
    import wfdb  # pip install wfdb

    record = wfdb.rdrecord("100", pn_dir="mitdb")
    freq_hz = record.fs
    value_data = record.p_signal.T[0]
    sig_name = record.sig_name[0]

    # WFDB doesn't have time information associated with this data, so create some.
    period_ns = (10 ** 9) // record.fs
    time_data = np.arange(value_data.size, dtype=np.int64) * period_ns

    # Remember start & end times for future query
    start_time_nano = 0
    end_time_nano = start_time_nano + (period_ns * value_data.size)

Define Signals and Sources
#############################

To define signals and sources in AtriumDB, you will use the `insert_measure` and `insert_device` methods of the AtriumSDK object.

.. code-block:: python

    # Define a new signal.
    new_measure_id = sdk.insert_measure(measure_tag=sig_name, freq=freq_hz, freq_units="Hz")

    # Define a new source.
    device_tag = "MITDB_record_100"
    new_device_id = sdk.insert_device(device_tag=device_tag)

Read and Write Data
#####################

To write and read data in AtriumDB, you will use the `write_data_easy` and `get_data` methods of the AtriumSDK object.

.. code-block:: python

    # Write Data
    sdk.write_data_easy(new_measure_id, new_device_id, time_data, value_data, freq_hz, freq_units="Hz")

    # Read Data
    _, read_time_data, read_value_data = sdk.get_data(measure_id=new_measure_id, start_time_n=start_time_nano, end_time_n=end_time_nano, device_id=new_device_id)
    assert np.array_equal(time_data, read_time_data)
    assert np.array_equal(value_data, read_value_data)


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