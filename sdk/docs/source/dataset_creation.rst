Creating a new dataset
####################################

The :ref:`AtriumSDK.create_dataset <create_dataset_label>` method is a powerful and flexible way to create a new dataset in AtriumDB. This method allows you to specify various options such as the type of metadata database to use, the protection mode, and the behavior when new data overlaps with existing data. In this section, we will explore the depths of the `AtriumSDK.create_dataset` method and its various options.

Parameters
----------

- **dataset_location**: A file path or a path-like object that points to the directory in which the dataset will be written. This is a required parameter.

- **database_type**: Specifies the type of metadata database to use. Options are "sqlite", "mysql", or "mariadb". The default is "sqlite".

- **protected_mode**: Specifies the protection mode of the metadata database. Allowed values are "True" or "False". If "True", data deletion will not be allowed. If "False", data deletion will be allowed. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.

- **overwrite**: Specifies the behavior to take when new data being inserted overlaps in time with existing data. Allowed values are "error", "ignore", or "overwrite". Upon triggered overwrite: if "error", an error will be raised. If "ignore", the new data will not be inserted. If "overwrite", the old data will be overwritten with the new data. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.

- **connection_params**: A dictionary containing connection parameters for "mysql" or "mariadb" database type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.

Examples
--------

If you anticipate that a single user will access your dataset and you prefer a simple setup, utilizing SQLite to power the metadata is the optimal choice. This option also ensures that the dataset remains portable, as SQLite files can be transferred like any regular file along with the binary data.

Create a new local dataset using the default SQLite to power the metadata db:

   .. code-block:: python

       from atriumdb import AtriumSDK

       sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset")

If you're hosting data for a large team or powering multiple visualizations, and your dataset is accessed concurrently by different consumers, it is advisable to use a more robust technology such as MariaDB to support AtriumDB's metadata.

Create a new local dataset using MariaDB:

   .. code-block:: python

       from atriumdb import AtriumSDK

       connection_params = {
           'host': "localhost",
           'user': "user",
           'password': "pass",
           'database': "new_dataset",
           'port': 3306
       }
       sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", connection_params=connection_params)

If you aim to safeguard your dataset from overwriting previous data, you can utilize the "overwrite='error'" parameter to raise an error instead of completing the write operation.

   .. code-block:: python

       from atriumdb import AtriumSDK
       import numpy as np

       sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="sqlite", overwrite="error")

       time_data = np.arange(10, dtype=np.int64)
       value_data = np.sin(time)

       # Works!
       sdk.write_data_easy(measure_id, device_id, time_data, value_data, 1, freq_units="Hz", time_units="s")

       # Error!
       sdk.write_data_easy(measure_id, device_id, time_data, value_data, 1, freq_units="Hz", time_units="s")
