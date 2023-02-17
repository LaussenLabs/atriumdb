Quick Start
-----------

.. toctree::
   :maxdepth: 2

This quick start guide will walk you through the process of creating a new dataset, pulling example data from WFDB, defining signals and sources, and reading and writing data in AtriumDB.

Dataset Location
#######################

To create a new dataset or continue working on a previous dataset, you will need to specify a local path where the dataset should be stored.

.. code-block:: python

    # Import AtriumSDK python object.
    from atriumdb import AtriumSDK

    # Define a directory path where the dataset will be stored.
    dataset_location = "./example_dataset"

    # Create AtriumSDK python object.
    sdk = AtriumSDK(dataset_location=dataset_location)

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
    time_data = np.arange(value_data) * period_ns

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
    sdk.write_data_easy(new_measure_id, new_device_id, time_data, value_data, freq_nhz)

    # Read Data
    _, read_time_data, read_value_data = sdk.get_data(measure_id=new_measure_id, start_time_n=start_time_nano, end_time_n=end_time_nano, device_id=new_device_id)
    assert np.array_equal(time_data, read_time_data)
    assert np.array_equal(value_data, read_value_data)
