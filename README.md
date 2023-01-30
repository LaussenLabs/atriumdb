# AtriumDB

AtriumDB the software (as distinct from AtriumDB the dataset), is currently broken into three main components: The SDK, The API and The WAL System.

## Installation

To install the development version, clone or download the repository.

```
$ git clone https://github.com/LaussenLabs/atriumdb
```

Navigate to the base directory and run:

```
$ cd atriumdb/sdk
$ pip install .
```
    
## Quick Start

### Dataset Location

Once installed, you can start creating a new dataset or continue working on a previous dataset simply by specifying a local path where the dataset should be stored.

```python
# Import AtriumSDK python object.
from atriumdb import AtriumSDK

# Define a directory path where the dataset will be stored.
dataset_location = "./example_dataset"

# Create AtriumSDK python object.
sdk = AtriumSDK(dataset_location=dataset_location)
```

### Pull Some Example Data from WFDB

```python
import numpy as np
import wfdb  # pip install wfdb

record = wfdb.rdrecord("100", pn_dir="mitdb")
freq_nhz = record.fs * 1_000_000_000
value_data = record.p_signal.T[0]
sig_name = record.sig_name[0]

# WFDB doesn't have time information associated with this data, so create some.
period_ns = (10 ** 9) // record.fs
time_data = np.arange(value_data) * period_ns

# Remember start & end times for future query
start_time_nano = 0
end_time_nano = start_time_nano + (period_ns * value_data.size)
```

### Define Signals and Sources

```python
# Define a new signal.
new_measure_id = 17
sdk.insert_measure(measure_tag=sig_name, freq_nhz=freq_nhz)

# Define a new source.
new_device_id = 100
device_tag = "MITDB_record_100"
sdk.insert_device(device_tag=device_tag)
```

### Read and Write Data

```python
# Write Data
sdk.write_data_easy(new_measure_id, new_device_id, time_data, value_data, freq_nhz)

# Read Data
_, read_time_data, read_value_data = sdk.get_data(new_measure_id, start_time_nano, end_time_nano,
                                                  device_id=new_device_id)
assert np.array_equal(time_data, read_time_data)
assert np.array_equal(value_data, read_value_data)
```

## Using MySQL or MariaDB
    
If you would like to use mysql/mariadb or you want to alter the location of the sqlite database, you must specify the URI to the database manually.

```python
# Specify database_uri for the metadata relational database.
username = 'user'
password = 'pass'
host = 'localhost'
db_name = 'example_database'
database_uri = "mysql+pymysql://{}:{}@{}/{}".format(username, password, host, db_name)

# Create AtriumSDK python object.
sdk = AtriumSDK(dataset_location=dataset_location, database_uri=database_uri)
```

### Generating The Docs

```console
$ cd sdk
$ sphinx-build -b html .\docs\source\ .\docs\build\ ; .\docs\make html
```

## The SDK

The most fundamental component of AtriumDB, the SDK contains the storage, indexing and compression/decompression components.

The SDK's main functionality is called through an `AtriumSDK` object, which can be setup to communicate with a variety of relational database technologies to access the block information metadata, or can be configured into a "remote mode" that utilizes the API to query meta information and raw signals.

The SDK can be broken up into three distinct processes, each have their own Class, stored as variables within the AtriumSDK object: `AtriumSql`, `AtriumFileHandler` and `BlockCodec`.

### Key Functions/Methods

Usage of the SDK, through the `AtriumSDK` object revolves around a few key methods.

`AtriumSDK` - The constructor of the sdk object, configures your connection to your desired dataset through a metadata db uri, a binary file location and a C DLL that facilitates the compression/decompression. Instead of a file location and metadata db uri, can take an API URL and Auth Token to activate "remote mode".

`AtriumSDK.get_data` - Queries data, from the dataset your SDK Object points at, based on measure_id (what signal you're looking for), device_id (the bedspace or device from which the data was recorded) and a start and end time in nanoseconds. The function returns two numpy arrays representing the time and value data, as well as a list of `BlockMetadata` header objects that contain metadata about each block of raw data accessed to complete the query

`AtriumSDK.write_data` - The complementary function to `get_data`, this function writes time and value data, to the dataset your SDK Object points at, indexed by measure_id and device_id.

`AtriumSDK.get_data_api` - Similar to `get_data`, instead using an sdk object configured in "remote mode". Will be implicitly called if `get_data` is called by an object in "remote mode".

`AtriumSDK.get_available_measures` - Returns a list of all measure_ids that have data in your dataset for a given device_id and optionally a time window.

`AtriumSDK.get_available_devices` - Returns a list of all device_ids that have data in your dataset for a given measure_id and optionally a time window.

`AtriumSDK.get_interval_array` - Returns an array of intervals of available data for a given measure_id, device_id and optionally a time window, and a "gap tolerance".

`AtriumSDK.get_combined_intervals` - The same as the above function, but takes a list of multiple measure_ids and returns all intervals where all measures are present.

`AtriumSDK.get_random_data` - Gets a random selection of a given measure_id.

## The API

This directory contains the code that implements an API server, facilitated through an SDK object to interact with a locally available dataset and FastAPI to interact with requests.

## The WAL System

The Write-Append Log (WAL), is used for streaming writes into an AtriumDB dataset. It uses `concurrent.futures` to concurrently create wal files from input messages and signal (by dropping the file pointer) a consumer manager/workers to read the wal file and ingest the data into the dataset using `write_data`.

## TSC Format

A general purpose compressed timeseries file format (Time Series Compression: TSC), built as a sequence of 1 or more "blocks" containing compressed time data, compressed value data and a block header with information about the raw data, compression/decompression methods and meta information about the compressed block itself.

## Metadata Schema

![schema](docs/SDK_Schema.png)

