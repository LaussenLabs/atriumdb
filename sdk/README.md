# AtriumDB
AtriumDB is a comprehensive solution for the management and analysis of physiological waveform data. It includes a powerful SDK for data compression, storage and retrieval.

Alongside regularly sampled waveforms, AtriumDB stores **aperiodic** signals (NIBP, labs,
anything with irregular timestamps) and **text/string** measures (alarm messages, ventilator
modes, clinical state markers, free-text notes) — see [Beyond waveforms](#beyond-waveforms-aperiodic-and-text-data).

## Installation
From PyPI (recommended)
```console
$ pip install atriumdb
```
This will install the base version of AtriumDB, allowing the reading and writing to local datasets, supported by sqlite3 only.
For more installation options including support to MariaDB datasets see the [documentation](https://docs.atriumdb.io/installation.html).
To install from source see GitHub readme [here](https://github.com/LaussenLabs/atriumdb).

## Quick Start

### Creating a new dataset
To create a new dataset, you can use the `create_dataset` method. This method allows you to specify the type of metadata database to use and where the data will be stored.
```python
from atriumdb import AtriumSDK

# Create a new local dataset using SQLite
sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="sqlite")

# OR create a new local dataset using MariaDB
connection_params = {
    'host': "localhost",
    'user': "user",
    'password': "pass",
    'database': "new_dataset",
    'port': 3306
}

sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", connection_params=connection_params)
```
The sdk object is how you will interact with the dataset including retrieving data, saving data and any of the other methods defined in the [documentation](https://docs.atriumdb.io/contents.html).

### Connecting to an existing dataset
To connect to an already created dataset, you will need to specify a local path where the dataset is stored if it's a sqlite database. 
If it's a MariaDB dataset you will also have to specify the connection parameters.

```python
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
```

## Beyond waveforms: aperiodic and text data

Every measure carries two independent pieces of metadata: a `signal_kind` (the temporal shape —
`waveform`, `sample`, `event` or `state`) and a `value_type` (`numeric` or `string`). Declare both
for anything that is not a regularly sampled waveform.

```python
import numpy as np

# An irregular numeric measure. Aperiodic measures still need a NOMINAL frequency
# (it becomes part of the measure's key); freq=0 is rejected.
nibp_id = sdk.insert_measure(measure_tag="NIBP_SYS", freq=1.0, freq_units="Hz",
                             units="mmHg", signal_kind="sample", value_type="numeric")
sdk.write_time_value_pairs(nibp_id, device_id,
                           np.array([0.0, 190.0, 480.0]), np.array([118.0, 122.0, 115.0]),
                           freq=1.0, freq_units="Hz", time_units="s")

# A text measure. 'state' keeps the values decodable inside windows; 'event' gives
# a presence-only channel.
mode_id = sdk.insert_measure(measure_tag="vent_mode", freq=1.0, freq_units="Hz",
                             units="mode", signal_kind="state", value_type="string")
sdk.write_time_value_pairs(mode_id, device_id, np.array([0.0, 300.0]), ["SIMV", "PRVC"],
                           freq=1.0, freq_units="Hz", time_units="s")

times, values = sdk.get_string_data(mode_id, start_time_n=0, end_time_n=10 ** 18,
                                    device_id=device_id)
```

Event queries turn `from → to` markers into state intervals, with explicit censoring flags:

```python
intervals = sdk.get_event_intervals(
    measure=mode_id, from_value="SIMV", to_value="PRVC",
    device_id=device_id, start_time=0, end_time=3600, time_units="s")
# [{'start_time_n': ..., 'end_time_n': ..., 'start_censored': False, 'end_censored': False}]
```

A `DatasetDefinition` can be anchored on those events (`anchor`, or `from`/`to`) to build
event-centred cohorts, and the windowing iterator rasterizes aperiodic and string measures onto
the window grid so every measure yields a fixed-length array. See the
[documentation](https://docs.atriumdb.io/) — *Measure Metadata*, *Event Queries*,
*Aperiodic and String Measure Windowing*, *Event-Anchored Regions*, and *Operations*.
