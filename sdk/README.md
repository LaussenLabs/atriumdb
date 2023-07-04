# AtriumDB
AtriumDB is a comprehensive solution for the management and analysis of physiological waveform data. It includes a powerful SDK for data compression, storage and retrieval.

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
