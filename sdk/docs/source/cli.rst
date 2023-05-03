##################
CLI
##################

****************
Overview
****************

The AtriumDB CLI is a powerful command-line interface tool designed to interact with AtriumDB datasets. It provides users with the ability to authenticate remote mode access, display quick information about the contents of AtriumDB datasets, and import/export data to and from AtriumDB in various popular formats such as CSV, Parquet, Numpy, and WFDB.

The CLI is easy to install and use, with comprehensive help documentation available through the ``atriumdb --help`` command.

Features
==============

- Authenticate remote mode access for secure data handling
- Display quick information about AtriumDB datasets, including measures, devices, and patients
- Import and export data to and from AtriumDB in various formats (CSV, Parquet, Numpy, WFDB)
- Filter and search data using various parameters (e.g., tags, names, units, frequency, identifiers)
- Support for cohort files to automatically configure export parameters
- Customizable export options, including packaging type and output location

Installation
==============

To install the AtriumDB CLI, simply run the following command:

.. code-block::

    pip install atriumdb[cli]

Once installed, you can access the CLI by running the `atriumdb` command. For help on usage, run:

.. code-block::

    atriumdb --help

**************************
Authentication
**************************

To use the CLI for authentication and remote access, you will need to install the `atriumdb` package with the `cli` and `remote` optional dependency.

.. code-block:: bash

    pip install atriumdb[cli,remote]

You can then use the `atriumdb` CLI to set the endpoint URL and log in to the remote API.

.. code-block:: bash

    atriumdb --endpoint-url http://example.com/api/v1 login

If the endpoint URL is already set in the .env file or as an environment variable, you can simply log in like this:

Create a file named `.env` in the same directory as your script and add the following content:

.. code-block:: ini

    ATRIUMDB_ENDPOINT_URL=http://example.com/api/v1

Now, you can log in using the CLI:

.. code-block:: bash

    atriumdb login

After logging in, the `atriumdb` CLI will store the API token in the `.env` file. You can update your `.env` file to include the API token as well:

.. code-block:: ini

    ATRIUMDB_ENDPOINT_URL=http://example.com/api/v1
    ATRIUMDB_API_TOKEN=4e78a93749ead7893

Now, you can access the remote dataset using the AtriumSDK object, as shown in the "Connecting to an Existing Dataset" section.


*********************************************************
Using AtriumSDK in Remote Mode with CLI Authentication
*********************************************************

Once you have successfully logged in using the AtriumDB CLI, as described in the `Authentication`_ section, you can use the AtriumSDK in remote mode with your Python scripts. The AtriumSDK will automatically detect the stored API token in the `.env` file when `metadata_connection_type` is set to `"api"`.

To use the AtriumSDK in remote mode, follow these steps:

1. Log in using the AtriumDB CLI, as described in the `Authentication`_ section.

2. Create a Python script in the same directory as the `.env` file containing the stored API token.

3. Import the `AtriumSDK` class from the `atriumdb` package:

.. code-block:: python

    from atriumdb import AtriumSDK

4. Instantiate the `AtriumSDK` object with the `metadata_connection_type` parameter set to `"api"`:

.. code-block:: python

    sdk = AtriumSDK(metadata_connection_type="api", api_url="http://example.com/api/v1")

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

**************************
Querying Metadata
**************************

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

**************************
Import / Export
**************************

The AtriumDB CLI provides the ability to import and export data between different AtriumDB datasets and various popular formats such as CSV, Parquet, Numpy, and WFDB. This chapter will cover the usage of the import and export commands, along with their supported options and parameters.

Export Command
==============

The ``export`` command allows you to transfer data from an AtriumDB dataset to another dataset or to various file formats. The command supports a range of options for specifying the data to be exported, the format, and the destination.

Here's the basic syntax for the ``export`` command:

.. code-block:: bash

    atriumdb export [OPTIONS]

The available options for the ``export`` command are:

- ``--format``: The format of the exported data. Supported formats are "adb", "csv", "parquet", "numpy", and "wfdb". Default is "adb".
- ``--packaging-type``: The type of packaging for the exported data. Supported types are "files", "tar", and "gzip". Default is "files".
- ``--cohort-file``: Path to a cohort file for automatically configuring export parameters.
- ``--measure-ids``: List of measure IDs to export.
- ``--measures``: List of measure tags to export.
- ``--device-ids``: List of device IDs to export.
- ``--devices``: List of device tags to export.
- ``--patient-ids``: List of patient IDs to export.
- ``--mrns``: List of MRNs to export.
- ``--start-time``: Start time for exporting data.
- ``--end-time``: End time for exporting data.
- ``--dataset-location-out``: Path to the export directory.
- ``--metadata-uri-out``: The URI of a metadata server.
- ``--database-type-out``: The metadata database type.
- ``--by-patient``: Whether or not to include patient mapping. Default is False.

Here's an example of using the ``export`` command to export data in CSV format:

.. code-block:: bash

    atriumdb export --format csv --dataset-location-out /path/to/export/directory

Import Command
==============

The ``import`` command is currently under development and will be available in a future release. It will allow users to import data into an AtriumDB dataset from various file formats.

For now, anything import could do you can do with export by switching the source and target datasets.

Cohort Files
============

Cohort files are a convenient way to specify a set of export parameters in a single file. The AtriumDB CLI supports YAML-formatted cohort files, which can be used with the ``--cohort-file`` option in the ``export`` command.

Here's an example of a cohort file:

.. code-block:: yaml

    measures:
      - HR
      - RR
    measure_ids:
      - 1
      - 2
    devices:
      - device_A
      - device_B
    device_ids:
      - 10
      - 11
    patient_ids:
      - 100
      - 101
    mrns:
      - 123456
      - 789012
    start_epoch_s: 1620000000
    end_epoch_s: 1620100000

To use a cohort file with the ``export`` command, simply provide the path to the file with the ``--cohort-file`` option:

.. code-block:: bash

    atriumdb export --cohort-file /path/to/cohort.yaml --dataset-location-out /path/to/export/directory

*********************************
List of Commands and Options
*********************************

This section provides an overview of the available commands and their respective options in the AtriumDB CLI.

AtriumDB Command
================
The atriumdb command is a command line interface for the Atrium database, allowing you to import and export data from the database. The import subcommand is used to import data into the database from common formats, while the export subcommand is used to export data from the database to common formats.

The atriumdb dataset is defined by the following environment variables or corresponding command line options. You can use command line options in place of the environment variables for a more flexible configuration.

Usage
-----

.. code-block:: bash

   atriumdb [options]

Options
-------

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


Login Command
=============

This command authenticates the user with the AtriumDB server using a QR code. It sends a request to the server to get the authentication configuration, generates a device code, displays a QR code for the user to scan, and then checks if the user has completed the authentication process. If successful, the API token is set in the user's environment variables.

Usage
-----

.. code-block:: bash

   atriumdb login

Options
-------

This command does not have any options.

Export Command
==============
This command exports data from AtriumDB to the specified format and packaging type. Users can filter the data to be exported using various options such as measure ids, device ids, patient ids, and MRNs. The export command also supports specifying a cohort file to automatically configure export parameters.

Usage
-----

.. code-block:: bash

   atriumdb export [options]

Options
-------

+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                 | Description                                                                                                                                                                       |
+========================+===================================================================================================================================================================================+
| --format               | Format of the exported data (default: adb). Choices: adb, csv, parquet, numpy, wfdb. Currently, only adb and csv formats are supported for export.                                |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --packaging-type       | Type of packaging for the exported data (default: files). Choices: files, tar, gzip.                                                                                              |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --cohort-file          | Cohort file for automatically configuring export parameters. Supported formats: .yml, .yaml.                                                                                      |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --measure-ids          | List of measure ids to export.                                                                                                                                                    |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --measures             | List of measure tags to export. Measure ids matching the tags will be added to the export list.                                                                                   |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --device-ids           | List of device ids to export.                                                                                                                                                     |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --devices              | List of device tags to export. Device ids matching the tags will be added to the export list.                                                                                     |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --patient-ids          | List of patient ids to export.                                                                                                                                                    |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --mrns                 | List of MRNs to export.                                                                                                                                                           |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --start-time           | Start time for exporting data in epoch seconds.                                                                                                                                   |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --end-time             | End time for exporting data in epoch seconds.                                                                                                                                     |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --dataset-location-out | Path to export directory. This option or the ATRIUMDB_EXPORT_DATASET_LOCATION environment variable must be specified.                                                             |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --metadata-uri-out     | The URI of a metadata server. If not specified, the ATRIUMDB_METADATA_URI_OUT environment variable will be used.                                                                  |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --database-type-out    | The metadata database type. If not specified, the ATRIUMDB_DATABASE_TYPE_OUT environment variable will be used.                                                                   |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --by-patient           | Whether or not to include patient mapping (default: False).                                                                                                                       |
+------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Import Command
==============

This command imports data to AtriumDB from various formats.

Usage
-----

.. code-block:: bash

   atriumdb import [options]

Options
-------

+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option               | Description                                                                                                                                                                       |
+======================+===================================================================================================================================================================================+
| --format             | Format of the imported data (default: adb). Choices: adb, csv, parquet, numpy, wfdb.                                                                                              |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --packaging-type     | Type of packaging for the imported data (default: files). Choices: files, tar, gzip.                                                                                              |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --dataset-location-in| Path to import directory.                                                                                                                                                         |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --metadata-uri-in    | The URI of a metadata server to import from.                                                                                                                                      |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --endpoint-url-in    | The endpoint to connect to for a remote AtriumDB server to import from.                                                                                                           |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --measure-ids        | List of measure ids to import.                                                                                                                                                    |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --measures           | List of measure tags to import.                                                                                                                                                   |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --device-ids         | List of device ids to import.                                                                                                                                                     |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --devices            | List of device tags to import.                                                                                                                                                    |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --patient-ids        | List of patient ids to import.                                                                                                                                                    |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --mrns               | List of MRNs to import.                                                                                                                                                           |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --start-time         | Start time for importing data.                                                                                                                                                    |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| --end-time           | End time for importing data.                                                                                                                                                      |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Measure Command
===============

The measure command is a group command for managing measures in a relational database. It has a subcommand `ls` which lists measures based on the provided search criteria.

Usage
-----

.. code-block:: bash

   atriumdb measure ls [options]

Options
-------

+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Option         | Description                                                                                                                              |
+================+==========================================================================================================================================+
| --tag-match    | Filters measures by matching the provided string against the measure's tag field. Only measures with a tag field containing the          |
|                | specified string will be returned.                                                                                                       |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| --name-match   | Filters measures by matching the provided string against the measure's name field. Only measures with a name field containing the        |
|                | specified string will be returned.                                                                                                       |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| --unit         | Filters measures by their units. Only measures with a unit field equal to the specified string will be returned.                         |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| --freq         | Filters measures by their frequency. Only measures with a frequency field equal to the specified value will be returned.                 |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| --freq-units   | Specifies the unit of frequency for the `--freq` option. The default unit is Hz.                                                         |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+
| --source-id    | Filters measures by their source identifier. Only measures with a source identifier field equal to the specified value will be returned. |
+----------------+------------------------------------------------------------------------------------------------------------------------------------------+

Device Command
==============

The device command is a group command for managing devices in the linked relational database. It has a subcommand called `ls` which lists devices in the linked relational database that match the specified search criteria, such as tag, name, manufacturer, model, bed ID, and source ID.

Usage
-----

.. code-block:: bash

   atriumdb device ls [options]

Options
-------

+----------------------+------------------------------------------------------------------------------------------+
| Option               | Description                                                                              |
+======================+==========================================================================================+
| --tag-match          | Filter devices by tag string match                                                       |
+----------------------+------------------------------------------------------------------------------------------+
| --name-match         | Filter devices by name string match                                                      |
+----------------------+------------------------------------------------------------------------------------------+
| --manufacturer-match | Filter devices by manufacturer string match                                              |
+----------------------+------------------------------------------------------------------------------------------+
| --model-match        | Filter devices by model string match                                                     |
+----------------------+------------------------------------------------------------------------------------------+
| --bed-id             | Filter devices by bed identifier                                                         |
+----------------------+------------------------------------------------------------------------------------------+
| --source-id          | Filter devices by source identifier                                                      |
+----------------------+------------------------------------------------------------------------------------------+

Patient Command
===============

This command group manages patient records in a healthcare database, it has a subcommand called `ls` which lists patient records with optional filters.
The command retrieves information about all patients in the linked relational database, including their id, medical record number (MRN), gender,
date of birth (DOB), first name, middle name, last name, first seen timestamp, last updated timestamp, and source identifier.

Usage
-----

.. code-block:: bash

   atriumdb patient ls [options]

Options
-------

+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| Option         | Description                                                                                                                                 |
+================+=============================================================================================================================================+
| --skip         | Number of patients to skip before starting to return the results. Useful for pagination.                                                    |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --limit        | Maximum number of patients to return in the result. Useful for pagination.                                                                  |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --age-years-min| Minimum age in years to filter patients. Filters patients whose age is greater than or equal to the specified value.                        |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --age-years-max| Maximum age in years to filter patients. Filters patients whose age is less than or equal to the specified value.                           |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --gender       | Filter patients based on their gender.                                                                                                      |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --source-id    | Filter patients by their source identifier. Useful for filtering patients from a specific data source.                                      |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --first-seen   | Filter patients by the timestamp when they were first seen, in epoch time. Filters patients whose first seen timestamp is greater than or   |
|                | equal to the specified value.                                                                                                               |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| --last-updated | Filter patients by the timestamp when their record was last updated, in epoch time. Filters patients whose last updated timestamp is        |
|                | greater than or equal to the specified value.                                                                                               |
+----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
