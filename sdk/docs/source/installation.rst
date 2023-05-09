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

AtriumDB with MariaDB
----------------------

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
MariaDB container running on port 3306 with the root user with the password example. You will use this username,
password, and port when instantiating an AtriumDB object later.

.. code-block:: bash

    $ docker run --name mariadb -d -p 127.0.0.1:3306:3306 -v /path/for/mariadb/data/on/host:/var/lib/mysql -e MARIADB_ROOT_PASSWORD='password' mariadb:latest