##################
The Atriumdb CLI
##################

****************
Overview
****************

The Atriumdb CLI is a powerful command-line interface tool designed to interact with Atriumdb datasets. It provides users with the ability to authenticate remote mode access, display quick information about the contents of Atriumdb datasets, and import/export data to and from Atriumdb in various popular formats such as CSV, Parquet, Numpy, and WFDB.

The CLI is easy to install and use, with comprehensive help documentation available through the `atriumdb --help` command.

Features
==============

- Authenticate remote mode access for secure data handling
- Display quick information about Atriumdb datasets, including measures, devices, and patients
- Import and export data to and from Atriumdb in various formats (CSV, Parquet, Numpy, WFDB)
- Filter and search data using various parameters (e.g., tags, names, units, frequency, identifiers)
- Support for cohort files to automatically configure export parameters
- Customizable export options, including packaging type and output location

Installation
==============

To install the Atriumdb CLI, simply run the following command:

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

You can also filter the results by using various options, such as `--tag-match`, `--name-match`, `--unit`, `--freq`, `--freq-units`, `--source-id`, `--manufacturer-match`, `--model-match`, `--bed-id`, `--gender`, `--age-years-min`, `--age-years-max`, `--first-seen`, and `--last-updated`.

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

*********************************
List of Commands and Options
*********************************

