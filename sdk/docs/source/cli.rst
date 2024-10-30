##################
CLI
##################

.. toctree::
   :maxdepth: 1

================
Overview
================

The AtriumDB CLI is a powerful command-line interface tool designed to interact with AtriumDB datasets. It provides users with the ability to authenticate remote mode access, display quick information about the contents of AtriumDB datasets, and import/export data to and from AtriumDB in various popular formats such as CSV, Parquet, Numpy, and WFDB.

The CLI is easy to install and use, with comprehensive help documentation available through the ``atriumdb --help`` command.

Features
------------------

- Authenticate remote mode access for secure data handling
- Display quick information about AtriumDB datasets, including measures, devices, and patients
- Import and export data to and from AtriumDB in various formats (CSV, Parquet, Numpy, WFDB)
- Filter and search data using various parameters (e.g., tags, names, units, frequency, identifiers)
- Support for cohort files to automatically configure export parameters
- Customizable export options, including packaging type and output location

Installation
-----------------

To install the AtriumDB CLI, simply run the following command:

.. code-block::

    pip install atriumdb[cli]

Once installed, you can access the CLI by running the `atriumdb` command. For help on usage, run:

.. code-block::

    atriumdb --help

=========================
Authentication
=========================

To use the AtriumDB CLI for authentication and remote access, you only need to specify the `endpoint-url` when logging in. The CLI will now automatically detect the URL from the last login for all subsequent commands.

To log in and set the endpoint URL, simply use the `atriumdb login` command with the `--endpoint-url` option. This option now supports login with an optional port:

.. code-block:: bash

    # Login with a port
    atriumdb login --endpoint-url "https://example.com:443/v1"

    # Login without specifying a port
    atriumdb login --endpoint-url "https://example.com/v1"

After logging in, the authentication credentials, including the API token and endpoint URL, are securely stored. If you need to refresh your token, you can now do so with the `atriumdb refresh-token` command, which uses the stored endpoint URL:

.. code-block:: bash

    atriumdb refresh-token

To view your current CLI configuration, including the endpoint URL and API token, use the `atriumdb config` command:

.. code-block:: bash

    atriumdb config

Authentication Timeout
-----------------------

If a certain period of inactivity passes or the CLI detects that the authentication has timed out, the CLI will prompt you to reauthenticate using the `atriumdb refresh-token` or re-run the `atriumdb login` command with your endpoint URL.

Modifying the `.env` File
---------------------------

Directly editing the `.env` file is no longer recommended. The AtriumDB CLI will manage all necessary environment variables for you, ensuring secure and effective handling of authentication credentials.

Connecting to an Existing Dataset
----------------------------------

After successfully logging in using the updated methods described above, you can continue to use the AtriumSDK in remote mode with any Python scripts. The synchronization with your current login state occurs automatically.

Remember that while instantiating the `AtriumSDK`, there's no need to explicitly provide the API token, as it will be read from the stored credentials:

.. code-block:: python

    sdk = AtriumSDK(metadata_connection_type="api", api_url="https://example.com/v1")

    # The SDK will now use the stored API token from the last successful login.

========================================================
Using AtriumSDK in Remote Mode with CLI Authentication
========================================================

Once you have successfully logged in using the AtriumDB CLI, as described in the `Authentication`_ section, you can use the AtriumSDK in remote mode with your Python scripts. The AtriumSDK will automatically detect the stored API token in the `.env` file when `metadata_connection_type` is set to `"api"`.

To use the AtriumSDK in remote mode, follow these steps:

1. Log in using the AtriumDB CLI, as described in the `Authentication`_ section.

2. Create a Python script in the same directory as the `.env` file containing the stored API token.

3. Import the `AtriumSDK` class from the `atriumdb` package:

.. code-block:: python

    from atriumdb import AtriumSDK

4. Instantiate the `AtriumSDK` object with the `metadata_connection_type` parameter set to `"api"`:

.. code-block:: python

    sdk = AtriumSDK(metadata_connection_type="api", api_url="http://example.com/v1")

By setting `metadata_connection_type` to `"api"`, the AtriumSDK will automatically detect and use the API token stored in the `.env` file for remote API calls (alternatively you can specify the token in the `token` parameter).

Now, you can use the AtriumSDK's methods to interact with the remote dataset. Here are some examples:

.. code-block:: python

    # List all devices
    devices = sdk.get_all_devices()
    print(devices)

    # List all measures
    measures = sdk.get_all_measures()
    print(measures)

    # Search for measures by frequency and units
    searched_measures = sdk.search_measures(freq=60, freq_units="Hz")
    print(searched_measures)

    # Get measure information by measure ID
    measure_info = sdk.get_measure_info(measure_id=1)
    print(measure_info)

For more information on using the AtriumSDK methods, refer to the provided Python functions in the `AtriumSDK` page.

========================================================
Querying Metadata
========================================================

The `atriumdb` CLI allows you to query metadata about measures, devices, and patients. This can be helpful when you want to quickly explore and understand the contents of your dataset.

To query metadata, use the following commands:

1. List measures: ``atriumdb measure ls``

.. code-block:: bash

    atriumdb measure ls

.. code-block:: none

    Measure ID  Tag           Name                       Frequency (nHz)  Code  Unit    Unit Label  Unit Code  Source ID
    -----------  ------------  ------------------------  ---------------  ----  ------  ----------  ---------  ----------
    1            ECG_I         Lead I ECG                500000000000     PQRST  mV      µV          µV         1
    2            ECG_II        Lead II ECG               250000000000     QRS    mV      µV          µV         1
    3            ABP           Arterial Blood Pressure   500000000000     T      mmHG    µV          µV         1

2. List devices: ``atriumdb device ls``

.. code-block:: bash

    atriumdb device ls

.. code-block:: none

    Device ID  Tag       Name          Manufacturer  Model            Type     Bed ID  Source ID
    ---------  ------    -----------   ------------  --------------   ------   ------  ----------
    1          monitor   HeartMonitor  Philips       IntelliVue MP70  monitor  101     1
    2          monitor   HeartMonitor  Philips       IntelliVue MP50  monitor  102     1
    3          monitor   HeartMonitor  GE Healthcare Dash 4000        monitor  103     1

3. List patients: ``atriumdb patient ls``

.. code-block:: bash

    atriumdb patient ls

.. code-block:: none

    id  mrn       gender  dob                 first_name  middle_name  last_name  first_seen      last_updated    source_id
    --  -------   ------  -----------------  ----------  -----------   ---------  -------------   -------------   ----------
    1   12345678  M       326054449000000000  John        Doe           Smith      1588358449000   1589358449000   2

You can also filter the results by using various options, such as `--tag-match`, `--name-match`, `--unit`, `--freq`.

For example, to filter measures by a specific tag or frequency, use the `--tag-match` or `--freq` options:

.. code-block:: bash

    atriumdb measure ls --tag-match "ECG" --freq 250

.. code-block:: none

    Measure ID  Tag           Name                Frequency (nHz)  Code  Unit    Unit Label  Unit Code  Source ID
    -----------  ------------  ------------------  ---------------  ----  ------  ----------  ---------  ----------
    1            ECG_I         Lead I ECG          500000000000     PQRST  mV      µV          µV         1
    2            ECG_II        Lead II ECG         250000000000     QRS    mV      µV          µV         1

To filter devices by a specific tag or manufacturer, use the `--tag-match` or `--manufacturer-match` options:

.. code-block:: bash

    atriumdb device ls --tag-match "monitor" --manufacturer-match "Philips"

.. code-block:: none

    Device ID  Tag       Name          Manufacturer  Model            Type     Bed ID  Source ID
    ---------  ------    -----------   ------------  --------------   ------   ------  ----------
    1          monitor   HeartMonitor  Philips       IntelliVue MP70  monitor  101     1
    2          monitor   HeartMonitor  Philips       IntelliVue MP50  monitor  102     1

========================================================
Import / Export
========================================================

The AtriumDB CLI provides the ability to export data between different AtriumDB datasets and various popular formats such as CSV, TSC, Numpy, and WFDB.

Import
------------

The ``import`` command enables you to transfer data into an AtriumDB dataset from various file formats.

The basic syntax for the ``import`` command is as follows:

.. code-block:: bash

    atriumdb import path/to/dataset/ [OPTIONS]

For a complete list of options available for the ``import`` command, refer to the :ref:`Import Command <import_header>` section under "List of Commands and Options."


Export
------------

The ``export`` command allows you to transfer data from an AtriumDB dataset to another dataset or to various file formats.

Here's the basic syntax for the ``export`` command:

.. code-block:: bash

    atriumdb export path/to/definition.yaml [OPTIONS]

`path/to/definition.yaml` is a required field, pointing to a valid :ref:`Dataset Definition File <definition_file_format>`

You can go to the :ref:`Export Command <export_header>` section under "List of Commands and Options" for a complete list of options available for the ``export`` command.

Here's an example of using the ``export`` command to export data in CSV format:

.. code-block:: bash

    atriumdb export my_dataset_definition.yaml --format csv --dataset-location-out /path/to/export/directory

Definition Files
-------------------

Dataset Definition files are a convenient way to specify a set of export parameters in a single file.

Visit the :ref:`Dataset Definition File Page <definition_file_format>` to understand how to create a valid file.

========================================================
List of Commands and Options
========================================================

This section provides an overview of the available commands and their respective options in the AtriumDB CLI.

The atriumdb command is a command line interface for the Atrium database, allowing you to import and export data from the database. The import subcommand is used to import data into the database from common formats, while the export subcommand is used to export data from the database to common formats.

The atriumdb dataset is defined by the following environment variables or corresponding command line options. You can use command line options in place of the environment variables for a more flexible configuration.

.. code-block:: bash

   atriumdb [options]

+-------------------+--------------------------------------------------------------------------------------------------------------------+
| Option            | Description                                                                                                        |
+===================+====================================================================================================================+
| --dataset-location| The local path to a dataset.                                                                                       |
+-------------------+--------------------------------------------------------------------------------------------------------------------+
| --metadata-uri    | The URI of a metadata server.                                                                                      |
+-------------------+--------------------------------------------------------------------------------------------------------------------+
| --database-type   | The type of metadata database supporting the dataset. Choices: sqlite, mariadb, mysql, api.                        |
+-------------------+--------------------------------------------------------------------------------------------------------------------+
| --endpoint-url    | The endpoint to connect to for a remote AtriumDB server.                                                           |
+-------------------+--------------------------------------------------------------------------------------------------------------------+
| --api-token       | A token to authorize API access.                                                                                   |
+-------------------+--------------------------------------------------------------------------------------------------------------------+


Login
----------

This command authenticates the user with the AtriumDB server using a QR code or Web link.
It sends a request to the server to get the authentication configuration, generates a device code, displays a QR code and a URL for the user to scan or follow,
and then checks if the user has completed the authentication process. If successful, the API token is set in the user's environment variables.

Usage
^^^^^

.. code-block:: bash

   atriumdb login --endpoint-url <url>


Options
^^^^^^^

+----------------+----------------------------------------------------------------------------------------+
| Option         | Description                                                                            |
+================+========================================================================================+
| --endpoint-url | The endpoint to connect to for a remote AtriumDB server. This option is required.      |
+----------------+----------------------------------------------------------------------------------------+

Refresh Token
--------------

This command allows users to refresh their API token using the stored endpoint URL. It automatically retrieves the endpoint URL from the stored configurations and uses it to refresh the user’s authentication token.

Usage
^^^^^^

.. code-block:: bash

   atriumdb refresh_token

No options are required to run this command. However, it is essential that the endpoint URL is previously set using the
`atriumdb login --endpoint-url <your_endpoint_url>` command. If the endpoint URL is not already set, the command will prompt
the user to execute the login command with an endpoint URL.


Config
--------------

This command displays the current configuration of the CLI. It shows the endpoint URL, API token, dataset location, metadata URI, and
database type that are currently set for the CLI session. If any of these configurations are not set,
it will display "Not Set" for that particular configuration.

The command also determines whether the CLI is operating in "Remote" or "Local" mode based on the database type
(marked as "api" for remote operation) and displays this mode as part of the configuration. If the API token is set,
only the first 15 characters of the API token will be displayed initially for security reasons, followed by the full API token.

Usage
^^^^^

.. code-block:: bash

   atriumdb config

This command does not require any options.


Note: The `config` command is particularly helpful for quickly verifying the current setup or diagnosing configuration-related issues.
It provides a clear, tabulated view of critical settings that affect how the CLI interacts with AtriumDB and other services.

.. _export_header:

Export
--------------
This command exports data from a dataset according to the specified parameters.
Users have the option to customize their data export in several ways, including gap tolerance, deidentification of patient data,
specifying patient columns, and others.

.. code-block:: bash

   atriumdb export <definition_filename> [OPTIONS]

Options
^^^^^^^

+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                    | Description                                                                                                                                                |
+===========================+============================================================================================================================================================+
| `<definition_filename>`   | Path to the dataset definition file. This is a required argument and not an option.                                                                        |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--gap-tolerance`         | The time (in --time-units units) defining the minimum gap between different sections of data.                                                              |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--deidentify`            | Whether to deidentify patient data. Accepts 'True', 'False', or a filename for custom ID mapping. Default is 'False'.                                      |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--patient-cols`          | List of patient columns to transfer. Multiple values are accepted.                                                                                         |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--block-size`            | The target number of values per compression block.                                                                                                         |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--include-labels`        | Whether to include labels in the data transfer. Default is True.                                                                                           |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--measure-tag-match-rule`| Determines how to match the measures by tags. Choices: 'all', 'best'.                                                                                      |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--time-shift`            | Amount of time (in the specified units) by which to shift all timestamps in the data.                                                                      |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--time-units`            | Units for time-related parameters like gap-tolerance and time-shift. Choices: 'ns', 'us', 'ms', 's'. Default is 's'.                                       |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--export-format`         | The format used for exporting data. Choices: 'tsc', 'csv', 'npz', 'wfdb'. Default is 'tsc'.                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--dataset-location-out`  | Path to export directory. This can also be set via the ATRIUMDB_EXPORT_DATASET_LOCATION environment variable.                                              |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--metadata-uri-out`      | The URI of a metadata server, relevant when exporting in 'tsc' format with a mariadb supported dataset. Can be set via ATRIUMDB_METADATA_URI_OUT variable. |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `--database-type-out`     | The metadata database type. Can also be specified through the ATRIUMDB_DATABASE_TYPE_OUT environment variable.                                             |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+