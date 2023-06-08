Installation
============

.. toctree::
   :maxdepth: 2

To install the development version of AtriumDB, you will need to clone or download the repository from GitHub.

.. code-block:: bash

    $ git clone https://github.com/LaussenLabs/atriumdb

Once you have the repository, navigate to the base directory and run the following command:

.. code-block:: bash

    $ cd atriumdb/sdk
    $ pip install .

When atriumdb is hosted on `PyPI <https://pypi.org/>`_, you can use:

.. code-block:: bash

    $ pip install atriumdb

Or with a wheel file:

.. code-block:: bash

    $ pip install atriumdb-0.1.0-py3-none-any.whl

Which will install the base version of atriumdb, allowing the reading and writing to local datasets,
supported by sqlite3 only.

Install Options
----------------

The AtriumDB package provides several optional dependencies that extend its capabilities beyond the base version. These optional dependencies are not required to run the core functions of AtriumDB but provide additional features that some users may find helpful.

MariaDB Support
^^^^^^^^^^^^^^^

AtriumDB can interact with datasets supported by MariaDB. To enable this feature, the `mariadb` optional dependency must be installed.

.. code-block:: bash

    $ pip install atriumdb[mariadb]

This will install `mariadb` version 1.1.6.

Remote Operations
^^^^^^^^^^^^^^^^^

To connect to and read datasets that are hosted remotely, you can install the `remote` optional dependency.

.. code-block:: bash

    $ pip install atriumdb[remote]

This will install the following packages:

- `requests` (version >= 2.28.2, < 3)
- `auth0-python` (version >= 4.1.0, < 5)
- `qrcodeT` (version >= 1.0.4, < 2)
- `python-dotenv` (version >= 0.21, < 1)

Command Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AtriumDB also supports command line import/export of data and command line authentication to authorize remote connections. This feature can be enabled by installing the `cli` optional dependency.

.. code-block:: bash

    $ pip install atriumdb[cli]

This will install the following packages:

- `click` (version >= 8.1.3, < 9)
- `pandas` (version >= 1.5, < 2)
- `tabulate` (version >= 0.9.0, < 1)
- `fastparquet` (version == 2023.2.0)
- `tqdm` (version >= 4.65.0, < 5)
- `python-dotenv` (version >= 0.21, < 1)

All Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to enable all the optional features of AtriumDB at once, you can install all the optional dependencies.

.. code-block:: bash

    $ pip install atriumdb[all]

This will install all the packages listed in the `mariadb`, `remote`, and `cli` optional dependencies.

MariaDB
---------------

If you are using AtriumDB with MariaDB on Linux you have to make sure you have MariaDB server version 10.11 installed
on your computer. On windows this step is unnecessary and you can go straight to building the wheel file and pip
installing AtriumDB.

.. code-block:: bash

    $ curl -sS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup | sudo bash -s -- --mariadb-server-version="mariadb-10.11"
    $ apt-get update && apt-get install -y libmariadb3 libmariadb-dev

Then you have to pip install build and build a wheel file for AtriumDB before pip installing AtriumDB.

.. code-block:: bash

    $ git clone https://github.com/LaussenLabs/atriumdb
    $ pip install build
    $ cd atriumdb/sdk
    $ python -m build
    $ pip install dist/atriumdb-0.1.0-py3-none-any.whl[mariadb]

MariaDB Docker Setup
^^^^^^^^^^^^^^^^^^^^^

To use AtriumDB with MariaDB you need a MariaDB server running and docker is the easiest way to do that. First pull the
MariaDB image from dockerhub.

.. code-block:: bash

    $ docker pull mariadb

Then specify where you want MariaDB's data to sit on your host machine in the command below and it will start up a
MariaDB container running on port 3306 with the root user with the password, "password". You will use this username,
password, and port when instantiating an AtriumDB object later.

.. code-block:: bash

    $ docker run --name mariadb -d -p 127.0.0.1:3306:3306 -v /path/for/mariadb/data/on/host:/var/lib/mysql -e MARIADB_ROOT_PASSWORD='password' mariadb:latest