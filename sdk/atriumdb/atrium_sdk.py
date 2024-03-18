# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
import warnings

import numpy as np

import threading
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.adb_functions import allowed_interval_index_modes, get_block_and_interval_data, condense_byte_read_list, \
    find_intervals, sort_data, yield_data, convert_to_nanoseconds, convert_to_nanohz, \
    convert_from_nanohz, ALLOWED_TIME_TYPES, collect_all_descendant_ids, get_best_measure_id
from atriumdb.block import Block, convert_intervals_to_gap_array
from atriumdb.block_wrapper import T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, BlockMetadataWrapper
from atriumdb.file_api import AtriumFileHandler
from atriumdb.helpers import shared_lib_filename_windows, shared_lib_filename_linux, protected_mode_default_setting, \
    overwrite_default_setting
from atriumdb.helpers.settings import ALLOWABLE_OVERWRITE_SETTINGS, PROTECTED_MODE_SETTING_NAME, OVERWRITE_SETTING_NAME, \
    ALLOWABLE_PROTECTED_MODE_SETTINGS
from atriumdb.block_wrapper import BlockMetadata
from atriumdb.intervals.intervals import Intervals
import time
import atexit
from pathlib import Path, PurePath
from multiprocessing import cpu_count
import sys
import os
from typing import Union, List, Tuple, Optional

from atriumdb.sql_handler.sql_constants import SUPPORTED_DB_TYPES
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler
from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.filtered_iterator import FilteredDatasetIterator
from atriumdb.windowing.random_access_iterator import RandomAccessDatasetIterator
from atriumdb.windowing.verify_definition import verify_definition

try:
    import requests
    from requests import Session
    from dotenv import load_dotenv, set_key
    from websockets.sync.client import connect
    from atriumdb.adb_remote import _validate_bearer_token
    import jwt

    REQUESTS_INSTALLED = True
except ImportError:
    REQUESTS_INSTALLED = False

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import logging

_LOGGER = logging.getLogger(__name__)

DEFAULT_META_CONNECTION_TYPE = 'sqlite'
time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}

# _LOGGER.basicConfig(
#     level=_LOGGER.debug,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         _LOGGER.StreamHandler()
#     ]
# )


class AtriumSDK:
    """
    .. _atrium_sdk_label:

    The Core SDK Object that represents a single dataset and provides methods to interact with it. If you are using API
    mode then once you are finished with the object call the close method to clean up all connections.

    :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the dataset will be written.
    :param str metadata_connection_type: Specifies the type of connection to use for metadata. Options are "sqlite", "mysql", "mariadb", or "api". Default "sqlite".
    :param dict connection_params: A dictionary containing connection parameters for "mysql" or "mariadb" connection type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.
    :param int num_threads: Specifies the number of threads to use when processing data.
    :param str api_url: Specifies the URL of the server hosting the API in "api" connection type.
    :param str token: An authorization token for the API in "api" connection type.
    :param str refresh_token: A token to refresh your authorization token if it expires while you are doing something. Only for the API in "api" connection type.
    :param bool validate_token: Do you want the sdk to check if your token is valid when the sdk object is created and during execution. If it is not valid it will attempt to use the refresh token to get you a new one. If false the sdk will not attempt to refresh your token at any point. Only for "api" connection type.
    :param str tsc_file_location: A file path pointing to the directory in which the TSC (time series compression) files are written for this dataset. Used to customize the TSC directory location, rather than using `dataset_location/tsc`.
    :param str atriumdb_lib_path: A file path pointing to the shared library (CDLL) that powers the compression and decompression. Not required for most users.
    :param bool no_pool: If true disables Mariadb connection pooling, instead using a new connection for each query.

    Examples:
    -----------
    Simple Usage:

    >>> from atriumdb import AtriumSDK
    >>> sdk = AtriumSDK(dataset_location="./example_dataset")

    Advanced Usage:

    >>> # MySQL/MariaDB Connection
    >>> metadata_connection_type = "mysql"
    >>> connection_params = {
    >>>     'host': "localhost",
    >>>     'user': "user",
    >>>     'password': "pass",
    >>>     'database': "your_dataset_name",
    >>>     'port': 3306
    >>> }
    >>> sdk = AtriumSDK(dataset_location="./example_dataset", metadata_connection_type=metadata_connection_type, connection_params=connection_params)

    >>> # Remote API Mode
    >>> api_url = "http://example.com/v1"
    >>> token = "4e78a93749ead7893"
    >>> refresh_token = "87d9gvss9wj4"
    >>> metadata_connection_type = "api"
    >>> sdk = AtriumSDK(metadata_connection_type=metadata_connection_type, api_url=api_url, token=token, refresh_token=refresh_token)
    """

    def __init__(self, dataset_location: Union[str, PurePath] = None, metadata_connection_type: str = None,
                 connection_params: dict = None, num_threads: int = None, api_url: str = None, token: str = None,
                 refresh_token=None, validate_token=True, tsc_file_location: str = None, atriumdb_lib_path: str = None,
                 no_pool=False):

        self.dataset_location = dataset_location

        # Set default metadata connection type if not provided
        metadata_connection_type = DEFAULT_META_CONNECTION_TYPE if \
            metadata_connection_type is None else metadata_connection_type

        self.metadata_connection_type = metadata_connection_type

        # Set number of threads to max available minus 2 or 1, whichever is greater, if not provided
        if num_threads is None:
            num_threads = max(cpu_count() - 2, 1)

        # Set the C DLL path based on the platform if not provided
        if atriumdb_lib_path is None:
            if sys.platform == "win32":
                shared_lib_filename = shared_lib_filename_windows
            else:
                shared_lib_filename = shared_lib_filename_linux

            this_file_path = Path(__file__)
            atriumdb_lib_path = this_file_path.parent.parent / shared_lib_filename

        # Initialize the block object with the C DLL path and number of threads
        self.block = Block(atriumdb_lib_path, num_threads)
        self.sql_handler = None

        # Handle SQLite connections
        if metadata_connection_type == 'sqlite':
            # Ensure dataset_location is provided for SQLite mode
            if dataset_location is None:
                raise ValueError("dataset location must be specified for sqlite mode")

            # Convert dataset_location to a Path object if it's a string
            if isinstance(dataset_location, str):
                dataset_location = Path(dataset_location)

            # Set the default tsc_file_location if not provided.
            if tsc_file_location is None:
                tsc_file_location = dataset_location / 'tsc'

            # Set the SQLite database file path and create its parent directory if it doesn't exist
            db_file = Path(dataset_location) / 'meta' / 'index.db'
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Initialize the SQLiteHandler with the database file path
            self.sql_handler = SQLiteHandler(db_file)
            self.mode = "local"
            self.file_api = AtriumFileHandler(tsc_file_location)
            self.settings_dict = self._get_all_settings()

        # Handle MySQL or MariaDB connections
        elif metadata_connection_type == 'mysql' or metadata_connection_type == 'mariadb':
            # Ensure at least one of the required parameters is provided
            if dataset_location is None and tsc_file_location is None:
                raise ValueError("One of dataset_location, tsc_file_location must be specified.")

            # Convert dataset_location to a Path object if it's a string
            if isinstance(dataset_location, str):
                dataset_location = Path(dataset_location)

            # Set the default tsc_file_location if not provided
            if tsc_file_location is None:
                tsc_file_location = dataset_location / 'tsc'

            # Import the MariaDBHandler class and extract connection parameters
            from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
            host = connection_params['host']
            user = connection_params['user']
            password = connection_params['password']
            database = connection_params['database']
            port = connection_params['port']

            # Initialize the MariaDBHandler with the connection parameters
            self.sql_handler = MariaDBHandler(host, user, password, database, port, no_pool=no_pool)
            self.mode = "local"
            self.file_api = AtriumFileHandler(tsc_file_location)
            self.settings_dict = self._get_all_settings()

        # Handle API connections
        elif metadata_connection_type == 'api':
            # Check if the necessary modules are installed for API connections
            if not REQUESTS_INSTALLED:
                raise ImportError("Remote mode not installed. Please install atriumdb with pip install atriumdb[remote]")

            self.mode = "api"
            self.api_url = api_url
            self.validate_token = validate_token
            # remove the leading http stuff and replace it with ws, also remove any trailing slashes
            self.ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://").rstrip('/')
            # make this variable so once connection is made in the thread it is available to the sdk object
            self.websock_conn = None

            # need this variable so when we refresh the token we know if a .env file was supplied and we should set the token key
            self.dot_env_loaded = False

            # Load API and refresh token from environment variables if not provided
            if token is None:
                load_dotenv(dotenv_path="./.env", override=True)
                try:
                    token = os.environ['ATRIUMDB_API_TOKEN']
                    self.dot_env_loaded = True
                except KeyError:
                    token = None
            if refresh_token is None:
                load_dotenv(dotenv_path="./.env", override=True)
                try:
                    refresh_token = os.environ['ATRIUMDB_API_REFRESH_TOKEN']
                except KeyError:
                    refresh_token = None

            self.token, self.refresh_token = token, refresh_token

            if self.validate_token:
                # send get request to the atriumdb api to get the info you need to validate the API token
                auth_config_response = requests.get(f'{self.api_url}/auth/cli/code')

                if auth_config_response.status_code != 200:
                    raise RuntimeError(f"Something went wrong when getting Auth info from the API. HTTP Error {auth_config_response.status_code}")

                # information returned from atriumdb API that we need for validation now and refreshing of the token later
                self.auth_config = auth_config_response.json()

                try:
                    # validate bearer token and get its expiry, if token is expired already refresh it
                    decoded_token = _validate_bearer_token(self.token, self.auth_config)
                    self.token_expiry = decoded_token['exp']
                except jwt.PyJWTError:
                    # if the token is invalid attempt to refresh it
                    self._refresh_token()
        else:
            raise ValueError("metadata_connection_type must be one of sqlite, mysql, mariadb or api")

        # Create these caches early in case they get used in the initial creation of the caches below.
        self._measures, self._devices, self._label_sets = {}, {}, {}

        # Initialize measures and devices if not in API mode
        if metadata_connection_type != "api":
            self._measures = self.get_all_measures()
            self._devices = self.get_all_devices()
            self._label_sets = self.get_all_label_names()

            # Lazy caching, cache only built if patient info requested later
            self._patients = {}

            # A dictionary of a list of matching ids in order of number of blocks (DESC) for each tag.
            self._measure_tag_to_ordered_id = {}

            # Create a dictionary to map measure information to measure IDs
            self._measure_ids = {}
            for measure_id, measure_info in self._measures.items():
                self._measure_ids[(measure_info['tag'], measure_info['freq_nhz'], measure_info['unit'])] = measure_id

            # Create a dictionary to map device tags to device IDs
            self._device_ids = {}
            for device_id, device_info in self._devices.items():
                self._device_ids[device_info['tag']] = device_id

            # Create a dictionary to map label type names to their IDs
            self._label_set_ids = {}
            for label_id, label_info in self._label_sets.items():
                self._label_set_ids[label_info['name']] = label_id

            # Dictionaries to map MRN to patient ID and patient ID to MRN for quick lookups.
            self._mrn_to_patient_id = {}
            self._patient_id_to_mrn = {}

        # register an atexit hook to close any open connections properly
        atexit.register(self.close)

    @classmethod
    def create_dataset(cls, dataset_location: Union[str, PurePath], database_type: str = None, protected_mode: str = None,
                       overwrite: str = None, connection_params: dict = None, no_pool=False):
        """
        .. _create_dataset_label:

        A class method to create a new dataset.

        :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the dataset will be written.
        :param str database_type: Specifies the type of metadata database to use. Options are "sqlite", "mysql", or "mariadb".
        :param str protected_mode: Specifies the protection mode of the metadata database. Allowed values are "True" or "False". If "True", data deletion will not be allowed. If "False", data deletion will be allowed. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.
        :param str overwrite: Specifies the behavior to take when new data being inserted overlaps in time with existing data. Allowed values are "error", "ignore", or "overwrite". Upon triggered overwrite: if "error", an error will be raised. If "ignore", the new data will not be inserted. If "overwrite", the old data will be overwritten with the new data. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.
        :param dict connection_params: A dictionary containing connection parameters for "mysql" or "mariadb" database type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.
        :param bool no_pool: If true disables Mariadb connection pooling, instead using a new connection for each query.

        :return: An initialized AtriumSDK object.
        :rtype: AtriumSDK

        Examples:

        >>> from atriumdb import AtriumSDK
        >>> protected_mode, overwrite = None, None  # Use default values from `sdk/atriumdb/helpers/config.toml`
        >>> sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="sqlite", protected_mode=protected_mode, overwrite=overwrite)

        >>> # MySQL/MariaDB Connection
        >>> connection_params = {
        >>>     'host': "localhost",
        >>>     'user': "user",
        >>>     'password': "pass",
        >>>     'database': "new_dataset",
        >>>     'port': 3306
        >>> }
        >>> sdk = AtriumSDK.create_dataset(dataset_location="./new_dataset", database_type="mysql", protected_mode="False", overwrite="error", connection_params=connection_params)
        """

        # Create Dataset Directory if it doesn't exist.
        dataset_location = Path(dataset_location)
        if dataset_location.is_file():
            raise ValueError("The dataset location given is a file.")
        elif not dataset_location.is_dir():
            dataset_location.mkdir(parents=True, exist_ok=True)

        # Set default parameters.
        database_type = 'sqlite' if database_type is None else database_type
        if database_type not in SUPPORTED_DB_TYPES:
            raise ValueError("db_type {} not in {}.".format(database_type, SUPPORTED_DB_TYPES))

        protected_mode = protected_mode_default_setting if protected_mode is None else protected_mode
        overwrite = overwrite_default_setting if overwrite is None else overwrite
        if overwrite not in ALLOWABLE_OVERWRITE_SETTINGS:
            raise ValueError(f"overwrite setting {overwrite} not in {ALLOWABLE_OVERWRITE_SETTINGS}")
        if protected_mode not in ALLOWABLE_PROTECTED_MODE_SETTINGS:
            raise ValueError(f"protected_mode setting {protected_mode} not in {ALLOWABLE_PROTECTED_MODE_SETTINGS}")

        # Create the database
        if database_type == 'sqlite':
            if dataset_location is None:
                raise ValueError("dataset location must be specified for sqlite mode")
            db_file = Path(dataset_location) / 'meta' / 'index.db'
            db_file.parent.mkdir(parents=True, exist_ok=True)
            SQLiteHandler(db_file).create_schema()

        elif database_type == 'mysql' or database_type == "mariadb":
            from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
            host = connection_params['host']
            user = connection_params['user']
            password = connection_params['password']
            database = connection_params['database']
            port = connection_params['port']
            MariaDBHandler(host, user, password, database, port).create_schema()

        sdk_object = cls(dataset_location=dataset_location, metadata_connection_type=database_type,
                         connection_params=connection_params, no_pool=no_pool)

        # Add settings
        sdk_object.sql_handler.insert_setting(PROTECTED_MODE_SETTING_NAME, str(protected_mode))
        sdk_object.sql_handler.insert_setting(OVERWRITE_SETTING_NAME, str(overwrite))

        sdk_object.settings_dict = sdk_object._get_all_settings()

        return sdk_object

    def get_data(self, measure_id: int = None, start_time_n: int = None, end_time_n: int = None,
                 device_id: int = None, patient_id=None, time_type=1, analog=True, block_info=None,
                 time_units: str = None, sort=True, allow_duplicates=True, measure_tag: str = None,
                 freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                 device_tag: str = None, mrn: int = None):
        """
        The method for querying data from the dataset, indexed by signal type (measure_id or measure_tag with freq and units),
        time (start_time_n and end_time_n), and data source (device_id, device_tag, patient_id, or mrn).

        If measure_id is None, measure_tag along with freq and units must not be None, and vice versa.
        Similarly, if device_id is None, device_tag must not be None, and if patient_id is None, mrn must not be None.

        :param int measure_id: The measure identifier. If None, measure_tag must be provided.
        :param int start_time_n: The start epoch in nanoseconds of the data you would like to query.
        :param int end_time_n: The end epoch in nanoseconds. The end time is not inclusive.
        :param int device_id: The device identifier. If None, device_tag must be provided.
        :param int patient_id: The patient identifier. If None, mrn must be provided.
        :param int time_type: The type of time returned. Time_type=1 for timestamps.
        :param bool analog: Convert value return type to analog signal.
        :param block_info: Custom block_info list to skip metadata table check.
        :param str time_units: Unit for the time array returned. Options: ["s", "ms", "us", "ns"].
        :param bool sort: Whether to sort the returned data by time.
        :param bool allow_duplicates: Allow duplicate times in returned data. Affects performance if false.
        :param str measure_tag: A short string identifying the signal. Required if measure_id is None.
        :param freq: The sample frequency of the signal. Helpful with measure_tag.
        :param str units: The units of the signal. Helpful with measure_tag.
        :param str freq_units: Units for frequency. Options: ["nHz", "uHz", "mHz",
            "Hz", "kHz", "MHz"] default "nHz".
        :param str device_tag: A string identifying the device. Exclusive with device_id.
        :param int mrn: Medical record number for the patient. Exclusive with patient_id.

        :rtype: Tuple[List[BlockMetadata], numpy.ndarray, numpy.ndarray]
        :returns: List of Block header objects, 1D numpy array for time data, 1D numpy array for value data.
        """
        # check that a correct unit type was entered
        time_units = "ns" if time_units is None else time_units

        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        # make sure time type is either 1 or 2
        if time_type not in ALLOWED_TIME_TYPES:
            raise ValueError("Time type must be in [1, 2]")

        # convert start and end time to nanoseconds
        start_time_n = int(start_time_n * time_unit_options[time_units])
        end_time_n = int(end_time_n * time_unit_options[time_units])

        if device_id is None and device_tag is not None:
            device_id = self.get_device_id(device_tag)

        if patient_id is None and mrn is not None:
            patient_id = self.get_patient_id(mrn)

        _LOGGER.debug("\n")
        start_bench_total = time.perf_counter()

        # If the data is from the api.
        if self.mode == "api":
            if measure_id is None:
                assert measure_tag is not None and freq is not None and units is not None, \
                    "Must provide measure_id or all of measure_tag, freq, units"
                measure_id = self.get_measure_id(measure_tag, freq, units, freq_units)
            return self._get_data_api(measure_id, start_time_n, end_time_n, device_id=device_id, patient_id=patient_id,
                                      time_type=time_type, analog=analog, sort=sort, allow_duplicates=allow_duplicates)

        # Check the measure
        if measure_id is None:
            assert measure_tag is not None, "One of measure_id, measure_tag must be specified."
            measure_id = get_best_measure_id(self, measure_tag, freq, units, freq_units)

        # If we don't already have the blocks
        if block_info is None:
            # Select all blocks from the block_index (sql table) that match params.
            start_bench = time.perf_counter()
            block_list = self.sql_handler.select_blocks(int(measure_id), int(start_time_n), int(end_time_n), device_id, patient_id)

            # Concatenate continuous byte intervals to cut down on total number of reads.
            read_list = condense_byte_read_list(block_list)

            # if no matching block ids
            if len(read_list) == 0:
                return [], np.array([]), np.array([])

            # Map file_ids to filenames and return a dictionary.
            file_id_list = [row[2] for row in read_list]
            filename_dict = self.get_filename_dict(file_id_list)
            end_bench = time.perf_counter()
            # print(f"DB query took {round((end_bench - start_bench) * 1000, 4)} ms")
            _LOGGER.debug(f"get filename dictionary  {(end_bench - start_bench) * 1000} ms")

        # If we already have the blocks
        else:
            block_list = block_info['block_list']
            filename_dict = block_info['filename_dict']

            # if no matching block ids
            if len(block_list) == 0:
                return [], np.array([]), np.array([])

        # Read and decode the blocks.
        headers, r_times, r_values = self.get_data_from_blocks(block_list, filename_dict, start_time_n,
                                                               end_time_n, analog, time_type, sort=False,
                                                               allow_duplicates=allow_duplicates)

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates)

        end_bench_total = time.perf_counter()
        _LOGGER.debug(f"Total get data call took {round(end_bench_total - start_bench_total, 2)}: {r_values.size} values")
        _LOGGER.debug(f"{round(r_values.size / (end_bench_total - start_bench_total), 2)} values per second.")

        # convert time data from nanoseconds to unit of choice
        if time_units != 'ns':
            r_times = r_times / time_unit_options[time_units]

        return headers, r_times, r_values

    def get_data_from_blocks(self, block_list, filename_dict, start_time_n, end_time_n, analog=True,
                             time_type=1, sort=True, allow_duplicates=True):
        """
        Retrieve data from blocks.

        This method reads data from the specified blocks, decodes it, and returns the headers, times, and values.

        :param list block_list: List of blocks to read data from.
        :param dict filename_dict: Dictionary containing file information.
        :param int start_time_n: Start time of the data to read.
        :param int end_time_n: End time of the data to read.
        :param bool analog: Whether the data is analog or not, defaults to True.
        :param int time_type: The time type returned to you. Time_type=1 is time stamps, which is what most people will
         want. Time_type=2 is gap array and should only be used by advanced users. Note that sorting will not work for
        time type 2 and you may receive more values than you asked for because of this.
        :param bool sort: Whether to sort the returned data by time.
        :param bool allow_duplicates: Whether to allow duplicate times in the sorted returned data if they exist. Does
        nothing if sort is false.
        :return: Tuple containing headers, times, and values.
        :rtype: tuple
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        # Start performance benchmark
        start_bench = time.perf_counter()

        # Condense the block list for optimized reading
        read_list = condense_byte_read_list(block_list)

        # Read the data from the files using the read list
        encoded_bytes = self.file_api.read_file_list(read_list, filename_dict)

        _LOGGER.debug(f"read from disk {round((time.perf_counter() - start_bench) * 1000, 4)} ms")

        # Extract the number of bytes for each block
        num_bytes_list = [row[5] for row in block_list]

        # Decode the data and separate it into headers, times, and values
        r_times, r_values, headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                              time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates)

        return headers, r_times, r_values

    def _get_data_api(self, measure_id: int, start_time_n: int, end_time_n: int, device_id: int = None,
                      patient_id: int = None, mrn: int = None, time_type=1, analog=True, sort=True,
                      allow_duplicates=True):

        params = {'start_time': start_time_n, 'end_time': end_time_n, 'measure_id': measure_id, 'device_id': device_id,
                  'patient_id': patient_id, 'mrn': mrn}
        # Request the block information
        block_info_list = self._request("GET", 'sdk/blocks', params=params)

        # Check if there are no blocks in the response
        if len(block_info_list) == 0:
            # Return empty arrays for headers, request times and request values
            return [], np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Get the number of bytes for each block
        num_bytes_list = [row['num_bytes'] for row in block_info_list]

        # tik = time.perf_counter()
        encoded_bytes = self._block_websocket_request(block_info_list)
        # print(f"Time for {len(block_info_list)} blocks over websocket: {round((time.perf_counter() - tik) * 1000, 4)} ms")
        # print(f"Mb/s is {(np.sum(num_bytes_list)/(time.perf_counter() - tik))/1_000_000}\n")\

        # Decode the concatenated bytes to get headers, request times and request values
        r_times, r_values, headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                              time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates)

        return headers, r_times, r_values

    def _block_websocket_request(self, block_info_list):

        # check if the api token will expire within 30 seconds and if so refresh it
        if self.validate_token and time.time() >= self.token_expiry - 30:
            # get new API token
            self._refresh_token()

        # If there is no websocket connection create it now
        if self.websock_conn is None:
            # connect to the websocket
            self._websocket_connect()

        # make a comma delimited string of all the blocks we want from the API
        block_ids = ','.join([str(row['id']) for row in block_info_list])
        self.websock_conn.send(block_ids)

        # wait for all the blocks to be sent. At the end the message 'Atriumdb_Done' will be sent so we can break out
        # of the receiving loop without closing the connection
        message_list = []
        for message in self.websock_conn:
            if message == 'Atriumdb_Done':
                break
            elif message == 'expired_token':
                # this should not happen since the sdk should refresh the token before it tries to send a request
                raise RuntimeError("API token has expired")

            message_list.append(message)

        # Concatenate the content of all messages
        encoded_bytes = np.concatenate([np.frombuffer(message, dtype=np.uint8) for message in message_list], axis=None)

        return encoded_bytes

    def write_data_easy(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq: int,
                        scale_m: float = None, scale_b: float = None, time_units: str = None, freq_units: str = None):
        """
        .. _write_data_easy_label:

        The simplified method for writing new data to the dataset.

        This method makes it easy to write new data to the dataset by taking care of unit conversions and data type
        handling internally. It supports various time units and frequency units for user convenience.

        Example usage:

            >>> import numpy as np
            >>> sdk = AtriumSDK(dataset_location="./example_dataset")
            >>> new_measure_id = 21
            >>> new_device_id = 21
            >>> # Create some time data.
            >>> freq_hz = 1
            >>> time_data = np.arange(1234567890, 1234567890 + 3600, dtype=np.int64)
            >>> # Create some value data of equal dimension.
            >>> value_data = np.sin(time_data)
            >>> sdk.write_data_easy(measure_id=new_measure_id,device_id=new_device_id,time_data=time_data,value_data=value_data,freq=freq_hz,time_units="s",freq_units="Hz")

        :param interval_index_mode:
        :param int measure_id: The measure identifier corresponding to the measures table in the linked
            relational database.
        :param int device_id: The device identifier corresponding to the devices table in the linked
            relational database.
        :param np.ndarray time_data: A 1D numpy array representing the time information of the data to be written.
        :param np.ndarray value_data: A 1D numpy array representing the value information of the data to be written.
        :param int freq: The sample frequency of the data to be written. If you want to use units
            other than the default (nanohertz), specify the desired unit using the "freq_units" parameter.
        :param float scale_m: A constant factor to scale digital data to transform it to analog (None if raw data
            is already analog). The slope (m) in y = mx + b
        :param float scale_b: A constant factor to offset digital data to transform it to analog (None if raw data
            is already analog). The y-intercept (b) in y = mx + b
        :param str time_units: The unit used for the time data which can be one of ["s", "ms", "us", "ns"]. If units
            other than nanoseconds are used, the time values will be converted to nanoseconds and then rounded to the
            nearest integer.
        :param str freq_units: The unit used for the specified frequency. This value can be one of ["nHz", "uHz", "mHz",
            "Hz", "kHz", "MHz"]. If you use extremely large values for this, it will be converted to nanohertz
            in the backend, and you may overflow 64-bit integers.
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Set default time and frequency units if not provided
        time_units = "ns" if time_units is None else time_units
        freq_units = "nHz" if freq_units is None else freq_units

        # Convert time_data to nanoseconds if a different time unit is used
        if time_units != "ns":
            time_data = convert_to_nanoseconds(time_data, time_units)

        # Convert frequency to nanohertz if a different frequency unit is used
        if freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)

        # Determine the raw time type based on the size of time_data and value_data
        if time_data.size == value_data.size:
            raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
        else:
            raw_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Determine the encoded time type
        encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Determine the raw and encoded value types based on the dtype of value_data
        if np.issubdtype(value_data.dtype, np.integer):
            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        # Call the write_data method with the determined parameters
        self.write_data(measure_id, device_id, time_data, value_data, freq, int(time_data[0]), raw_time_type=raw_t_t,
                        raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                        scale_m=scale_m, scale_b=scale_b)

    def write_data(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq_nhz: int,
                   time_0: int, raw_time_type: int = None, raw_value_type: int = None, encoded_time_type: int = None,
                   encoded_value_type: int = None, scale_m: float = None, scale_b: float = None,
                   interval_index_mode: str = None, gap_tolerance: int = 0):
        """
        .. _write_data_label:

        Advanced method for writing new data to the dataset. This method can be used to express time data as a gap array
        (even sized array, odd values are indices of value_data after a gap and even values are the durations of the
        corresponding gaps in nanoseconds).

        :param int measure_id: Measure identifier corresponding to the measures table in the linked relational database.
        :param int device_id: Device identifier corresponding to the devices table in the linked relational database.
        :param numpy.ndarray time_data: 1D numpy array representing the time information of the data to be written.
        :param numpy.ndarray value_data: 1D numpy array representing the value information of the data to be written.
        :param int freq_nhz: Sample frequency, in nanohertz, of the data to be written.
        :param int time_0: Start time of the data to be written.
        :param int raw_time_type: Identifier representing the time format being written, corresponding to the options
            written in the block header.
        :param int raw_value_type: Identifier representing the value format being written, corresponding to the
            options written in the block header.
        :param int encoded_time_type: Identifier representing how the time information is encoded, corresponding
            to the options written in the block header.
        :param int encoded_value_type: Identifier representing how the value information is encoded, corresponding
            to the options written in the block header.
        :param float scale_m: Constant factor to scale digital data to transform it to analog (None if raw data
            is already analog). The slope (m) in y = mx + b
        :param float scale_b: Constant factor to offset digital data to transform it to analog (None if raw data
            is already analog). The y-intercept (b) in y = mx + b
        :param str interval_index_mode: Determines the mode for writing data to the interval index. Modes include "disable",
            "fast", and "merge". "disable" mode yields the fastest writing speed but loses lookup ability via the
            `AtriumSDK.get_interval_array` method. "fast" mode writes to the interval index in a non-optimized form,
            potentially creating multiple entries where one should exist, significantly increasing database size. "merge" mode
            consolidates intervals into single entries, maintaining a smaller table size but can incur a speed penalty,
            if the data inserted has lots of gaps, is aperiodic or isn't the newest data for that device-measure combination.
            For live data ingestion, "merge" is recommended.
        :param int gap_tolerance: The maximum number of nanoseconds that can occur between two consecutive values before
            it is treated as a break in continuity or gap in the interval index.

        :rtype: Tuple[numpy.ndarray, List[BlockMetadata], numpy.ndarray, str]
        :returns: A numpy byte array of the compressed blocks.
            A list of BlockMetadata objects representing the binary block headers.
            A 1D numpy array representing the byte locations of the start of each block.
            The filename of the written blocks.

        Examples:

            >>> import numpy as np
            >>> from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, \
            ...     V_TYPE_INT64, V_TYPE_DELTA_INT64
            >>> sdk = AtriumSDK(dataset_location="./example_dataset")
            >>> measure_id = 21
            >>> device_id = 21
            >>> freq_nhz = 1_000_000_000
            >>> time_zero_nano = 1234567890_000_000_000
            >>> gap_arr = np.array([42, 1_000_000_000, 99, 2_000_000_000])
            >>> value_data = np.sin(np.linspace(0, 4, num=200))
            >>> sdk.write_data(
            >>>     measure_id, device_id, gap_arr, value_data, freq_nhz, time_zero_nano,
            >>>     raw_time_type=T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
            >>>     raw_value_type=V_TYPE_INT64,
            >>>     encoded_time_type=T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
            >>>     encoded_value_type=V_TYPE_DELTA_INT64)
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Ensure there is data to be written
        assert value_data.size > 0, "Cannot write no data."

        # Ensure time data is of integer type
        assert np.issubdtype(time_data.dtype, np.integer), "Time information must be encoded as an integer."

        # Set default interval index and ensure valid type.
        interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode
        assert interval_index_mode in allowed_interval_index_modes, \
            f"interval_index must be one of {allowed_interval_index_modes}"

        # Calculate new intervals
        write_intervals = find_intervals(freq_nhz, raw_time_type, time_data, time_0, int(value_data.size))

        # check overwrite setting
        if OVERWRITE_SETTING_NAME not in self.settings_dict:
            raise ValueError("Overwrite behavior not set. Please set it in the sql settings table")
        overwrite_setting = self.settings_dict[OVERWRITE_SETTING_NAME]

        # Initialize variables for handling overwriting
        overwrite_file_dict, old_block_ids, old_file_list = None, None, None

        # if overwrite is ignore there is no reason to calculate this stuff
        if overwrite_setting != 'ignore':
            write_intervals_o = Intervals(write_intervals)

            # Get current intervals
            current_intervals = self.get_interval_array(
                measure_id, device_id=device_id, gap_tolerance_nano=0,
                start=int(write_intervals[0][0]), end=int(write_intervals[-1][-1]))

            current_intervals_o = Intervals(current_intervals)

            # Check if there is an overlap between current and new intervals
            if current_intervals_o.intersection(write_intervals_o).duration() > 0:
                _LOGGER.debug(f"Overlap measure_id {measure_id}, device_id {device_id}, "
                              f"existing intervals {current_intervals}, new intervals {write_intervals}")

                # Handle overwriting based on the overwrite_setting
                if overwrite_setting == 'overwrite':
                    _LOGGER.debug(
                        f"({measure_id}, {device_id}): value_data: {value_data} \n time_data: {time_data} \n write_intervals: {write_intervals} \n current_intervals: {current_intervals}")
                    overwrite_file_dict, old_block_ids, old_file_list = self._overwrite_delete_data(
                        measure_id, device_id, time_data, time_0, raw_time_type, value_data.size, freq_nhz)
                elif overwrite_setting == 'error':
                    raise ValueError("Data to be written overlaps already ingested data.")
                elif overwrite_setting == 'ignore':
                    pass
                else:
                    raise ValueError(f"Overwrite setting {overwrite_setting} not recognized.")

        # check if the write data will make at least one full block and if there will be a small block at the end
        num_full_blocks = value_data.size // self.block.block_size
        if (num_full_blocks > 0 and value_data.size % self.block.block_size != 0 and
                (raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO or raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO)):
            byte_start_array, encoded_bytes, encoded_headers = self.block.make_oversized_block(
                encoded_time_type, encoded_value_type, freq_nhz, num_full_blocks, raw_time_type, raw_value_type,
                scale_b,
                scale_m, time_0, time_data, value_data)
        # if all blocks are perfectly sized or there is less than one optimal block worth of data
        else:
            # Encode the blocks
            encoded_bytes, encoded_headers, byte_start_array = self.block.encode_blocks(
                time_data, value_data, freq_nhz, time_0,
                raw_time_type=raw_time_type,
                raw_value_type=raw_value_type,
                encoded_time_type=encoded_time_type,
                encoded_value_type=encoded_value_type,
                scale_m=scale_m,
                scale_b=scale_b)

        # Write the encoded bytes to disk
        filename = self.file_api.write_bytes(measure_id, device_id, encoded_bytes)

        # Use the header data to create rows to be inserted into the block_index and interval_index SQL tables
        block_data, interval_data = get_block_and_interval_data(
            measure_id, device_id, encoded_headers, byte_start_array, write_intervals,
            interval_gap_tolerance=gap_tolerance)

        # If data was overwritten
        if overwrite_file_dict is not None:
            # Add new data to SQL insertion data
            overwrite_file_dict[filename] = (block_data, interval_data)

            # Update SQL
            old_file_ids = [file_id for file_id, filename in old_file_list]
            _LOGGER.debug(
                f"{measure_id}, {device_id}): overwrite_file_dict: {overwrite_file_dict}\n "
                f"old_block_ids: {old_block_ids}\n old_file_ids: {old_file_ids}\n")
            self.sql_handler.update_tsc_file_data(overwrite_file_dict, old_block_ids, old_file_ids, gap_tolerance)

            # Delete old files
            # for file_id, filename in old_file_list:
            #     file_path = Path(self.file_api.to_abs_path(filename, measure_id, device_id))
            #     file_path.unlink(missing_ok=True)
        else:
            # Insert SQL rows
            self.sql_handler.insert_tsc_file_data(filename, block_data, interval_data, interval_index_mode, gap_tolerance)

        return encoded_bytes, encoded_headers, byte_start_array, filename

    def get_measure_id(self, measure_tag: str, freq: Union[int, float], units: str = None, freq_units: str = None):
        """
        .. _get_measure_id_label:

        Returns the identifier for a measure specified by its tag, frequency, units, and frequency units.

        :param str measure_tag: The tag of the measure.
        :param float freq: The frequency of the measure.
        :param str units: The unit of the measure (default is an empty string).
        :param str freq_units: The frequency unit of the measure (default is 'nHz').
        :return: The identifier of the measure.
        :rtype: int

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> measure_tag = "Temperature Measure"
        >>> freq = 100.0
        >>> units = "Celsius"
        >>> freq_units = "Hz"
        >>> sdk.get_measure_id(measure_tag, freq, units, freq_units)
        ... 7
        >>> measure_tag = "Measure That Does Not Exist."
        >>> sdk.get_measure_id(measure_tag, freq, units, freq_units)
        ... None
        """
        # Set default values for units and freq_units if not provided
        units = "" if units is None else units
        freq_units = "nHz" if freq_units is None else freq_units

        # Convert frequency to nanohertz
        freq_nhz = convert_to_nanohz(freq, freq_units)

        # If metadata connection type is "api", use API method to get the measure ID
        if self.metadata_connection_type == "api":
            return self._api_get_measure_id(measure_tag, freq_nhz, units, freq_units)

        # If measure ID is already in the cache, return it
        if (measure_tag, freq_nhz, units) in self._measure_ids:
            return self._measure_ids[(measure_tag, freq_nhz, units)]

        # Query the database for the measure ID
        row = self.sql_handler.select_measure(measure_tag=measure_tag, freq_nhz=freq_nhz, units=units)

        # If no row is found, return None
        if row is None:
            return None

        # Extract measure ID from the row and store it in the cache
        measure_id = row[0]
        self._measure_ids[(measure_tag, freq_nhz, units)] = measure_id

        # Return the measure ID
        return measure_id

    def _api_get_measure_id(self, measure_tag: str, freq: Union[int, float], units: str = None,
                            freq_units: str = None):
        params = {'measure_tag': measure_tag, 'freq': freq, 'unit': units, 'freq_units': freq_units}
        measure_result = self._request("GET", "measures/", params=params)

        units = "" if units is None else units

        for measure_id, measure_info in measure_result.items():
            tag_bool = measure_tag == measure_info['tag']
            freq_bool = freq == measure_info['freq_nhz']
            units_bool = measure_info['unit'] is None or units == measure_info['unit']
            if tag_bool and freq_bool and units_bool:
                return int(measure_id)
        return None

    def get_measure_info(self, measure_id: int):
        """
        .. _get_measure_info_label:

        Retrieve information about a specific measure in the linked relational database.

        :param int measure_id: The identifier of the measure to retrieve information for.

        :return: A dictionary containing information about the measure, including its id, tag, name, sample frequency
            (in nanohertz), code, unit, unit label, unit code, and source_id.
        :rtype: dict

        >>> # Connect to example_dataset
        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>>
        >>> # Retrieve information for measure with id=1
        >>> measure_id = 1
        >>> measure_info = sdk.get_measure_info(measure_id)
        >>> # print(measure_info)
        {
            'id': 1,
            'tag': 'Heart Rate',
            'name': 'Heart rate in beats per minute',
            'freq_nhz': 1000000000,
            'code': 'HR',
            'unit': 'BPM',
            'unit_label': 'beats per minute',
            'unit_code': 264864,
            'source_id': 1
        }
        """
        # Check if metadata connection type is API
        if self.metadata_connection_type == "api":
            return self._request("GET", f"measures/{measure_id}")

        # If measure_id is already in the cache, return the cached measure info
        if measure_id in self._measures:
            return self._measures[measure_id]

        # Query the SQL database for the measure information
        row = self.sql_handler.select_measure(measure_id=measure_id)

        # If no row is found, return None
        if row is None:
            return None

        # Unpack the row tuple into variables
        measure_id, measure_tag, measure_name, measure_freq_nhz, measure_code, measure_unit, measure_unit_label, \
        measure_unit_code, measure_source_id = row

        # Create a dictionary containing the measure information
        measure_info = {
            'id': measure_id,
            'tag': measure_tag,
            'name': measure_name,
            'freq_nhz': measure_freq_nhz,
            'code': measure_code,
            'unit': measure_unit,
            'unit_label': measure_unit_label,
            'unit_code': measure_unit_code,
            'source_id': measure_source_id
        }

        # Cache the measure information in the _measures dictionary
        self._measures[measure_id] = measure_info

        # Return the measure information dictionary
        return measure_info

    def search_measures(self, tag_match=None, freq=None, unit=None, name_match=None, freq_units=None):
        """
        .. _search_measures_label:

        Retrieve information about all measures in the linked relational database that match the specified search criteria.

        This function filters the measures based on the provided search criteria and returns a dictionary containing
        information about each matching measure, including its id, tag, name, sample frequency (in nanohertz), code, unit,
        unit label, unit code, and source_id.

        :param tag_match: A string to match against the `measure_tag` field. If not None, only measures with a `measure_tag`
            field containing this string will be returned.
        :type tag_match: str, optional
        :param freq: A value to match against the `measure_freq_nhz` field. If not None, only measures with a
            `measure_freq_nhz` field equal to this value will be returned.
        :type freq: int, optional
        :param unit: A string to match against the `measure_unit` field. If not None, only measures with a `measure_unit`
            field equal to this string will be returned.
        :type unit: str, optional
        :param name_match: A string to match against the `measure_name` field. If not None, only measures with a
            `measure_name` field containing this string will be returned.
        :type name_match: str, optional
        :param freq_units: The units for the freq parameter. (Default: "Hz")
        :type freq_units: str, optional
        :return: A dictionary containing information about each measure that matches the specified search criteria.
        :rtype: dict
        """
        # Check the metadata connection type and call the appropriate API search method if necessary
        if self.metadata_connection_type == "api":
            params = {'measure_tag': tag_match, 'freq': freq, 'unit': unit, 'measure_name': name_match,
                      'freq_units': freq_units}
            return self._request("GET", "measures/", params=params)

        # Set the default frequency units to "Hz" if not provided
        freq_units = "Hz" if freq_units is None else freq_units

        # Convert the frequency to nanohertz if necessary
        if freq_units != "nHz" and freq is not None:
            freq = convert_to_nanohz(freq, freq_units)

        # Get all measures from the database
        all_measures = self.get_all_measures()

        # Initialize the result dictionary
        result = {}

        # Iterate through all measures and filter them based on the search criteria
        for measure_id, measure_info in all_measures.items():
            # Create a list of boolean values for each search criterion
            match_bool_list = [
                tag_match is None or tag_match in measure_info['tag'],
                freq is None or freq == measure_info['freq_nhz'],
                unit is None or unit == measure_info['unit'],
                name_match is None or name_match in measure_info['name']
            ]

            # If all search criteria match, add the measure to the result dictionary
            if all(match_bool_list):
                result[measure_id] = measure_info

        # Return the filtered measures as a dictionary
        return result

    def get_all_measures(self):
        """
        .. _get_all_measures_label:

        Retrieve information about all measures in the linked relational database.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> all_measures = sdk.get_all_measures()
        >>> # print(all_measures)
        {1: {'id': 1,
             'tag': 'Heart Rate',
             'name': 'Heart Rate Measurement',
             'freq_nhz': 500,
             'code': 'HR',
             'unit': 'BPM',
             'unit_label': 'Beats per Minute',
             'unit_code': 'BPM',
             'source_id': 1},
         2: {'id': 2,
             'tag': 'Respiration Rate',
             'name': 'Respiration Rate Measurement',
             'freq_nhz': 500,
             'code': 'RR',
             'unit': 'BPM',
             'unit_label': 'Breaths per Minute',
             'unit_code': 'BPM',
             'source_id': 1}}

        :return: A dictionary containing information about each measure, including its id, tag, name, sample frequency
            (in nanohertz), code, unit, unit label, unit code, and source_id.
        :rtype: dict
        """
        # Check if connection type is API and call the appropriate method
        if self.metadata_connection_type == "api":
            measure_dict = self._request("GET", "measures/")
            return {int(measure_id): measure_info for measure_id, measure_info in measure_dict.items()}

        # Get all measures from the SQL handler
        measure_tuple_list = self.sql_handler.select_all_measures()

        # Initialize an empty dictionary to store measure information
        measure_dict = {}

        # Iterate through the list of measures and construct a dictionary for each measure
        for measure_info in measure_tuple_list:
            measure_id, measure_tag, measure_name, measure_freq_nhz, measure_code, \
            measure_unit, measure_unit_label, measure_unit_code, measure_source_id = measure_info

            # Add the measure information to the dictionary
            measure_dict[measure_id] = {
                'id': measure_id,
                'tag': measure_tag,
                'name': measure_name,
                'freq_nhz': measure_freq_nhz,
                'code': measure_code,
                'unit': measure_unit,
                'unit_label': measure_unit_label,
                'unit_code': measure_unit_code,
                'source_id': measure_source_id
            }

        return measure_dict

    def get_measure_id_list_from_tag(self, measure_tag: str, approx=True, freq=None, units=None, freq_units=None):
        """
        Returns a list of matching measure_ids for a given tag in DESC order by number of stored blocks.
        Helpful for finding all ids or the most prevalent id for a given tag. Optionally filters by frequency and units.

        :param str measure_tag: The tag of the measure.
        :param bool approx: If True, approximates the result based on first 100,000 rows of the block table.
            If False, queries the entire block table.
        :param freq: Optional frequency to filter measures.
        :param units: Optional units of the measure to filter by.
        :param freq_units: Units of the provided frequency. Converts frequency to nanohertz if not already.
        :return: A list of measure_ids
        """
        # Convert frequency to nanohertz if necessary
        if freq and freq_units and freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        # Get initial list of measure IDs from tag
        measure_ids = self._measure_tag_to_ordered_id.get(measure_tag, [])
        if not measure_ids:
            # Reload the cache if not found
            self._measure_tag_to_ordered_id = self.sql_handler.get_tag_to_measure_ids_dict(approx=approx)
            measure_ids = self._measure_tag_to_ordered_id.get(measure_tag, [])

        # Filter measure_ids by frequency and units if necessary
        if freq is not None or units is not None:
            filtered_measure_ids = []
            for measure_id in measure_ids:
                measure_info = self.get_measure_info(measure_id)
                if freq is not None and measure_info.get('freq_nhz') != freq:
                    continue
                if units is not None and measure_info.get('unit') != units:
                    continue
                filtered_measure_ids.append(measure_id)
            measure_ids = filtered_measure_ids

        return measure_ids

    def insert_measure(self, measure_tag: str, freq: Union[int, float], units: str = None, freq_units: str = None,
                       measure_name: str = None, measure_id: int = None, code: str = None, unit_label: str = None,
                       unit_code: str = None, source_id: int = None, source_name: str = None):
        """
        .. _insert_measure_label:

        Defines a new signal type to be stored in the dataset, as well as defining metadata related to the signal.

        `measure_tag`, `freq` and `units` are required information.

        >>> # Define a new signal with additional metadata.
        >>> freq = 500
        >>> freq_units = "Hz"
        >>> measure_tag = "ECG Lead II - 500 Hz"
        >>> measure_name = "Electrocardiogram Lead II Configuration 500 Hertz"
        >>> units = "mV"
        >>> code = "A0001"
        >>> unit_label = "millivolts"
        >>> unit_code = "mV01"
        >>> source_id = 123
        >>> measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units,
                                            measure_name=measure_name, code=code, unit_label=unit_label,
                                            unit_code=unit_code, source_id=source_id)

        :param str measure_tag: A short string identifying the signal.
        :param freq: The sample frequency of the signal.
        :param str units: The units of the signal.
        :param str optional freq_units: The unit used for the specified frequency. This value can be one of ["Hz",
            "kHz", "MHz"]. Keep in mind if you use extremely large values for this it will be
            converted to Hertz in the backend, and you may overflow 64bit integers.
        :param str optional measure_name: A long form description of the signal.
        :param int optional measure_id: The desired measure_id.
        :param str optional code: A specific code identifying the signal.
        :param str optional unit_label: A label for the unit.
        :param str optional unit_code: A code for the unit.
        :param int optional source_id: An identifier for the data source.
        :param str source_name: The name of the data source associated with the measure, used if source_id is not
            provided (optional).

        :return: The measure_id of the inserted or existing measure.
        :rtype: int

        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Check if measure_tag, measure_name, and units are either strings or None
        assert isinstance(measure_tag, str)
        assert isinstance(measure_name, str) or measure_name is None
        assert isinstance(units, str) or units is None
        assert isinstance(code, str) or code is None
        assert isinstance(unit_label, str) or unit_label is None
        assert isinstance(unit_code, str) or unit_code is None

        # Set default frequency unit to "nHz" if not provided
        freq_units = "nHz" if freq_units is None else freq_units
        units = "" if units is None else units

        # Handle source_name to source_id conversion
        if source_name and not source_id:
            source_id = self.get_source_id(source_name)
            if source_id is None:
                raise ValueError(f"Source name {source_name} not found.")

        # Convert frequency to nanohertz if the provided frequency unit is not "nHz"
        if freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)

        # Check for id clash
        if measure_id is not None:
            assert isinstance(measure_id, int)
            measure_info = self.get_measure_info(measure_id)
            if measure_info is not None:
                if measure_info['tag'] == measure_tag and \
                        measure_info['freq_nhz'] == freq and \
                        measure_info['unit'] == units:
                    return measure_id
                raise ValueError(f"Inserted measure_id {measure_id} already exists with data: {measure_info}")

        # Check if the measure already exists in the dataset
        check_measure_id = self.get_measure_id(measure_tag, freq, units=units)
        if check_measure_id is not None:
            return check_measure_id

        # Insert the new measure into the database
        inserted_measure_id = self.sql_handler.insert_measure(
            measure_tag, freq, units, measure_name, measure_id=measure_id, code=code, unit_label=unit_label,
            unit_code=unit_code, source_id=source_id)

        if inserted_measure_id is None:
            return inserted_measure_id

        # Add new measure_id into cache
        measure_info = {
            'id': inserted_measure_id,
            'tag': measure_tag,
            'name': measure_name,
            'freq_nhz': freq,
            'code': code,
            'unit': units,
            'unit_label': unit_label,
            'unit_code': unit_code,
            'source_id': source_id
        }
        self._measure_ids[(measure_tag, freq, units)] = inserted_measure_id
        self._measures[inserted_measure_id] = measure_info

        return inserted_measure_id

    def get_device_id(self, device_tag: str) -> int:
        """
        .. _get_device_id_label:

        Retrieve the identifier of a device in the linked relational database based on its tag. Or None if device
        not found.

        :param str device_tag: The tag of the device to retrieve the identifier for.

        :return: The identifier of the device. Or None if device not found.
        :rtype: int

        >>> # Connect to example_dataset
        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>>
        >>> # Retrieve the identifier of the device with tag "Monitor A1"
        >>> device_tag = "Monitor A1"
        >>> device_id = sdk.get_device_id(device_tag)
        >>> # print(device_id)
        ... 1
        """
        # Check if the metadata connection type is API
        if self.metadata_connection_type == "api":
            devices_result = self._request("GET", "devices/", params={'device_tag': device_tag})

            for device_id, device_info in devices_result.items():
                if device_tag == device_info['tag']:
                    return int(device_id)
            return None

        # If the device tag is already in the cached device IDs dictionary, return the cached ID
        if device_tag in self._device_ids:
            return self._device_ids[device_tag]

        # If the device tag is not in the cache, query the database using the SQL handler
        row = self.sql_handler.select_device(device_tag=device_tag)

        # If the device tag is not found in the database, return None
        if row is None:
            return None

        # If the device tag is found in the database, store the ID in the cache and return it
        device_id = row[0]
        self._device_ids[device_tag] = device_id
        return device_id

    def get_device_info(self, device_id: int):
        """
        .. _get_device_info_label:

        Retrieve information about a specific device in the linked relational database. Or None if device not found.

        :param int device_id: The identifier of the device to retrieve information for.

        :return: A dictionary containing information about the device, including its id, tag, name, manufacturer, model,
                 type, bed_id, and source_id. Or None if Device not found.
        :rtype: dict

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> device_id = 1
        >>> device_info = sdk.get_device_info(device_id)
        >>> # print(device_info)
        {'id': 1,
         'tag': 'Device A1',
         'name': 'Philips Device A1 in Room 1A',
         'manufacturer': 'Philips',
         'model': 'A1',
         'type': 'Device',
         'bed_id': 1,
         'source_id': 1}

        """
        # Check if metadata is fetched using API and call the appropriate method
        if self.metadata_connection_type == "api":
            return self._request("GET", f"devices/{device_id}")

        # If device info is already cached, return it
        if device_id in self._devices:
            return self._devices[device_id]

        # Fetch device info from the SQL database
        row = self.sql_handler.select_device(device_id=device_id)

        # If device not found in the database, return None
        if row is None:
            return None

        # Unpack the fetched row into individual variables
        device_id, device_tag, device_name, device_manufacturer, device_model, device_type, device_bed_id, \
        device_source_id = row

        # Create a dictionary with the device information
        device_info = {
            'id': device_id,
            'tag': device_tag,
            'name': device_name,
            'manufacturer': device_manufacturer,
            'model': device_model,
            'type': device_type,
            'bed_id': device_bed_id,
            'source_id': device_source_id,
        }

        # Cache the device information for future use
        self._devices[device_id] = device_info

        # Return the device information dictionary
        return device_info

    def search_devices(self, tag_match=None, name_match=None):
        """
        Retrieve information about all devices in the linked relational database that match the specified search criteria.
        This method supports searching by device tag and/or device name.

        :param tag_match: A string to match against the `device_tag` field. If not None, only devices with a `device_tag`
            field containing this string will be returned. Default is None.
        :type tag_match: str, optional
        :param name_match: A string to match against the `device_name` field. If not None, only devices with a `device_name`
            field containing this string will be returned. Default is None.
        :type name_match: str, optional
        :return: A dictionary containing information about each device that matches the specified search criteria, including
            its id, tag, name, manufacturer, model, type, bed_id, and source_id.
        :rtype: dict
        """
        # Check if the metadata connection type is "api" and call the appropriate method
        if self.metadata_connection_type == "api":
            return self._request("GET", "devices/", params={'device_tag': tag_match, 'device_name': name_match})

        # Get all devices from the linked relational database
        all_devices = self.get_all_devices()

        # Initialize an empty dictionary to store the search results
        result = {}

        # Iterate through all devices and their information
        for device_id, device_info in all_devices.items():
            # Create a list of boolean values to determine if the device matches the search criteria
            match_bool_list = [
                tag_match is None or tag_match in device_info['tag'],
                name_match is None or name_match in device_info['name']
            ]

            # If all conditions in the match_bool_list are True, add the device to the result dictionary
            if all(match_bool_list):
                result[device_id] = device_info

        # Return the dictionary containing the search results
        return result

    def get_all_devices(self):
        """
        .. _get_all_devices_label:

        Retrieve information about all devices in the linked relational database.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> all_devices = sdk.get_all_devices()
        >>> # print(all_devices)
        {1: {'id': 1,
             'tag': 'Monitor A1',
             'name': 'Philips Monitor A1 in Room 2A',
             'manufacturer': 'Philips',
             'model': 'A1',
             'type': 'Monitor',
             'bed_id': 2,
             'source_id': 1},
         2: {'id': 2,
             'tag': 'Monitor A2',
             'name': 'LG Monitor A2 in Room 2B',
             'manufacturer': 'LG',
             'model': 'A2',
             'type': 'Monitor',
             'bed_id': 2,
             'source_id': 2}}

        :return: A dictionary containing information about each device, including its id, tag, name, manufacturer,
            model, type, bed_id, and source_id.
        :rtype: dict
        """
        # Check if the metadata connection type is API
        if self.metadata_connection_type == "api":
            device_dict = self._request("GET", "devices/")
            return {int(device_id): device_info for device_id, device_info in device_dict.items()}

        # If the connection type is not API, use the SQL handler to get all devices
        device_tuple_list = self.sql_handler.select_all_devices()

        # Initialize an empty dictionary to store device information
        device_dict = {}

        # Iterate through the device tuple list
        for device_id, device_tag, device_name, device_manufacturer, device_model, device_type, device_bed_id, \
            device_source_id in device_tuple_list:
            # Create a dictionary for each device with its details
            device_dict[device_id] = {
                'id': device_id,
                'tag': device_tag,
                'name': device_name,
                'manufacturer': device_manufacturer,
                'model': device_model,
                'type': device_type,
                'bed_id': device_bed_id,
                'source_id': device_source_id,
            }

        # Return the dictionary containing all devices and their information
        return device_dict

    def insert_device(self, device_tag: str, device_name: str = None, device_id: int = None, manufacturer: str = None,
                      model: str = None, device_type: str = None, bed_id: int = None, bed_name: str = None,
                      source_id: int = None, source_name: str = None):
        """
        Insert a new device into the dataset and define its metadata.

        This method defines a new device to be stored in the dataset, specifying
        metadata such as the device's tag, name, manufacturer, model, type, and
        associations with a bed and source either by ID or by name. The `device_tag`
        is a required parameter, while all others are optional. If both an ID and a
        name are provided for a bed or source, the ID takes precedence.

        If the device_id is specified and already exists in the dataset with a
        different device_tag, a ValueError is raised. If `bed_name` or `source_name`

        is provided but does not match any existing records, a ValueError is also raised.

        Example usage:

        >>> # Define a new device using IDs.
        >>> device_tag = "Monitor A3"
        >>> device_name = "Philips Monitor A3 in Room 2B"
        >>> manufacturer = "Philips"
        >>> model = "A3"
        >>> device_type = "static"
        >>> bed_id = 102
        >>> source_id = 2
        >>> new_device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name,
                                              manufacturer=manufacturer, model=model, device_type=device_type,
                                              bed_id=bed_id, source_id=source_id)

        >>> # Define a new device using names.
        >>> bed_name = "Bed 2B"
        >>> source_name = "Source A"
        >>> new_device_id = sdk.insert_device(device_tag="Monitor B4", device_name="Siemens Monitor B4 in Bed 2B",
                                              manufacturer="Siemens", model="B4", device_type="dynamic",
                                              bed_name=bed_name, source_name=source_name)

        :param str device_tag: A unique string identifying the device (required).
        :param str device_name: A long form description of the device (optional).
        :param int device_id: Desired device_id, if specified, must not conflict with existing entries (optional).
        :param str manufacturer: The device's manufacturer (optional).
        :param str model: The device's model (optional).
        :param str device_type: The type of the device, either 'static' or 'dynamic' (optional).
        :param int bed_id: The ID of the bed associated with the device (optional).
        :param str bed_name: The name of the bed associated with the device, used if bed_id is not provided (optional).
        :param int source_id: The ID of the data source associated with the device (optional).
        :param str source_name: The name of the data source associated with the device, used if source_id is not provided (optional).

        :return: The device_id of the inserted or existing device.
        :rtype: int

        Raises:
            ValueError: If specified device_id already exists with a different device_tag.
                        If bed_name or source_name is provided but does not match any existing records.
        """
        # Handle source_name to source_id conversion
        if source_name and not source_id:
            source_id = self.get_source_id(source_name)
            if source_id is None:
                raise ValueError(f"Source name {source_name} not found.")

        # Handle bed_name to bed_id conversion
        if bed_name and not bed_id:
            bed_id = self.get_bed_id(bed_name)
            if bed_id is None:
                raise ValueError(f"Bed name {bed_name} not found.")

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Check for id clash
        if device_id is not None:
            assert isinstance(device_id, int)
            device_info = self.get_device_info(device_id)
            if device_info is not None:
                if device_info['tag'] == device_tag:
                    return device_id
                raise ValueError(f"Inserted device_id {device_id} already exists with data: {device_info}")

        # Check if the device_tag already exists in the dataset
        check_device_id = self.get_device_id(device_tag)
        if check_device_id is not None:
            # If it exists, return the existing device_id
            return check_device_id

        # If the device_tag does not exist, insert the new device using the sql_handler
        return self.sql_handler.insert_device(device_tag, device_name, device_id, manufacturer, model, device_type,
                                              bed_id, source_id)

    def get_patient_id(self, mrn: int):
        """
        Retrieve the patient ID associated with a given medical record number (MRN).

        This method looks for a patient's ID using their MRN. If the patient ID is not found in the initial search,
        it triggers a refresh of all patient data and searches again.

        :param int mrn: The medical record number for the patient, as an integer.
        :return: The patient ID as an integer if the patient is found; otherwise, None.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> patient_id = sdk.get_patient_id(mrn=123456)
        >>> print(patient_id)
        1
        """

        # Convert mrn to int for sql lookup
        mrn = int(mrn)

        # Check if we are in API mode
        if self.metadata_connection_type == "api":
            return self._request("GET", f"/patients/mrn|{mrn}", params={'time': None})['id']

        if mrn in self._mrn_to_patient_id:
            return self._mrn_to_patient_id[mrn]

        self.get_all_patients()

        if mrn in self._mrn_to_patient_id:
            return self._mrn_to_patient_id[mrn]

        return None

    def get_mrn(self, patient_id):
        """
        Retrieve the medical record number (MRN) associated with a given patient ID.

        This method searches for a patient's MRN using their patient ID. If the MRN is not found in the initial search,
        it triggers a refresh of all patient data and searches again.

        :param patient_id: The numeric identifier for the patient.
        :return: The MRN as an integer if the patient is found; otherwise, None.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> mrn = sdk.get_mrn(patient_id=1)
        >>> print(mrn)
        123456
        """
        # Check if we are in API mode
        if self.metadata_connection_type == "api":
            return self._request("GET", f"/patients/id|{patient_id}", params={'time': None})['mrn']

        if patient_id in self._patient_id_to_mrn:
            return self._patient_id_to_mrn[patient_id]

        self.get_all_patients()

        if patient_id in self._patient_id_to_mrn:
            return self._patient_id_to_mrn[patient_id]

        return None

    def get_patient_info(self, patient_id: int = None, mrn: int = None, time: int = None, time_units: str = None):
        """
        Retrieve information about a specific patient using either their numeric patient id or medical record number (MRN).

        :param int patient_id: The numeric identifier for the patient.
        :param int mrn: The medical record number for the patient.
        :param int time: (Optional) If you want the patient information for a specific time enter the epoch timestamp here.
         The function will get you the closest information available at a time less than or equal to the timestamp you
         provide. If left as None the function will get the most recent information.
        :param str time_units: (Optional) Units for the time. Valid options are 'ns', 's', 'ms', and 'us'. Default is nanoseconds.
        :return: A dictionary containing the patient's information, including id, MRN, gender, date of birth (dob),
                 first name, middle name, last name, date first seen, last updated datetime, source identifier, height, and weight.
                 If a time is specified you will also get the height/weight units and the time that each measurement was taken.
                 Returns None if patient not found.

        :raises ValueError: If both patient_id and mrn are not provided or neither of them are provided.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> patient_info = sdk.get_patient_info(patient_id=1)
        >>> print(patient_info)
        {
            'id': 1,
            'mrn': 123456,
            'gender': 'M',
            'dob': 946684800000000000,  # Nanoseconds since epoch
            'first_name': 'John',
            'middle_name': 'A',
            'last_name': 'Doe',
            'first_seen': 1609459200000000000,  # Nanoseconds since epoch
            'last_updated': 1609545600000000000,  # Nanoseconds since epoch
            'source_id': 1,
            'weight': 10.1,
            'weight_units': 'kg',
            'weight_time': 1609545500000000000,  # Nanoseconds since epoch
            'height': 50.0,
            'height_units': 'kg',
            'height_time': 1609544500000000000,  # Nanoseconds since epoch
        }
        """
        # Handle time units and conversion to nanoseconds
        if time_units and time:
            if time_units not in time_unit_options.keys():
                raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")
            time *= time_unit_options[time_units]

        # Check if we have either patient ID or MRN
        if patient_id is None and mrn is None:
            raise ValueError("Either patient_id or mrn must be provided.")
        # make sure they supply only one of patient id or mrn
        if patient_id is not None and mrn is not None:
            raise ValueError("Only one of patient_id or mrn should be provided.")

        # Check if we are in API mode
        if self.metadata_connection_type == "api":
            if patient_id is not None:
                return self._request("GET", f"/patients/id|{patient_id}", params={'time': time})
            return self._request("GET", f"/patients/mrn|{mrn}", params={'time': time})

        patient_info = None

        # Try getting the patient by MRN from the cache
        if mrn is not None:
            # Convert mrn to int for proper sql lookup
            mrn = int(mrn)
            if mrn in self._mrn_to_patient_id:
                patient_id = self._mrn_to_patient_id[mrn]

        # Try getting the patient by ID from the cache
        if patient_id is not None and patient_id in self._patients:
            patient_info = self._patients[patient_id]

        # If we did not find the patient, refresh the patient cache
        if patient_info is None or patient_id is None:
            self.get_all_patients()

        # Try finding the patient in the updated cache if necessary
        if patient_info is None and mrn is not None and mrn in self._mrn_to_patient_id:
            patient_id = self._mrn_to_patient_id[mrn]

        if patient_info is None and patient_id is not None and patient_id in self._patients:
            patient_info = self._patients[patient_id]

        # If the patient is still not found, return None
        if patient_info is None or patient_id is None:
            return None

        # If a time was specified then get the patient info closest to that timestamp
        if time is not None:
            # make them none incase no matching info is available for the supplied time
            patient_info['height'], patient_info['height_units'], patient_info['height_time'] = None, None, None
            patient_info['weight'], patient_info['weight_units'], patient_info['weight_time'] = None, None, None

            # update the patient dictionary with the height/weight closest to the time
            height = self.sql_handler.select_closest_patient_history(patient_id=patient_id, field='height', time=time)
            if height:
                patient_info['height'], patient_info['height_units'], patient_info['height_time'] = height[3], height[4], height[5]
            weight = self.sql_handler.select_closest_patient_history(patient_id=patient_id, field='weight', time=time)
            if weight:
                patient_info['weight'], patient_info['weight_units'], patient_info['weight_time'] = weight[3], weight[4], weight[5]
        return patient_info

    def get_all_patients(self, skip=None, limit=None):
        """
        .. _get_all_patients_label:

        Retrieve information about all patients in the linked relational database.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> all_patients = sdk.get_all_patients()
        >>> # print(all_patients)
        {1: {'id': 1,
             'mrn': 123456,
             'gender': 'M',
             'dob': 946684800000000000,
             'first_name': 'John',
             'middle_name': 'A',
             'last_name': 'Doe',
             'first_seen': 1609459200000000000,
             'last_updated': 1609545600000000000,
             'source_id': 1,
             'weight': 10.1,
             'height': 50.0},
         2: {'id': 2,
             'mrn': 654321,
             'gender': 'F',
             'dob': 978307200000000000,
             'first_name': 'Jane',
             'middle_name': 'B',
             'last_name': 'Smith',
             'first_seen': 1609642000000000000,
             'last_updated': 1609728400000000000,
             'source_id': 1,
             'weight': 9.12,
             'height': 43.2}}

        :return: A dictionary containing information about each patient, including their id, mrn, gender, dob,
            first_name, middle_name, last_name, first_seen, last_updated, source_id, height and weight.
        :rtype: dict
        """
        # Check if the metadata connection type is API and call the appropriate method
        if self.metadata_connection_type == "api":
            return self._api_get_all_patients(skip=skip, limit=limit)

        # Retrieve all patient records from the database
        patient_tuple_list = self.sql_handler.select_all_patients()

        # Set default values for skip and limit if not provided
        skip = 0 if skip is None else skip
        limit = len(patient_tuple_list) if limit is None else limit

        # Initialize an empty dictionary to store patient information
        patient_dict = {}

        # Iterate over the patient records and populate the patient_dict
        for patient_id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height in \
                patient_tuple_list[skip:skip + limit]:
            patient_dict[patient_id] = {
                'id': patient_id,
                'mrn': mrn,
                'gender': gender,
                'dob': dob,
                'first_name': first_name,
                'middle_name': middle_name,
                'last_name': last_name,
                'first_seen': first_seen,
                'last_updated': last_updated,
                'source_id': source_id,
                'weight': weight,
                'height': height
            }

        # Cache the results
        self._patients = patient_dict

        # Create a dictionary to map MRN to patient ID and patient ID to MRN for quick lookups.
        self._mrn_to_patient_id = {}
        self._patient_id_to_mrn = {}
        for patient_id, patient_info in self._patients.items():
            mrn = patient_info['mrn']
            if mrn is None:
                continue
            self._mrn_to_patient_id[mrn] = patient_id
            self._patient_id_to_mrn[patient_id] = mrn

        # Return the populated patient_dict
        return patient_dict

    def _api_get_all_patients(self, skip=None, limit=None):
        skip = 0 if skip is None else skip

        if limit is None:
            limit = 100
            patient_dict = {}
            while True:
                result_temp = self._request("GET", "patients/", params={'skip': skip, 'limit': limit})
                result_dict = {int(patient_id): patient_info for patient_id, patient_info in result_temp.items()}

                if len(result_dict) == 0:
                    break
                patient_dict.update(result_dict)
                skip += limit
        else:
            result_temp = self._request("GET", "patients/", params={'skip': skip, 'limit': limit})
            patient_dict = {int(patient_id): patient_info for patient_id, patient_info in result_temp.items()}

        return patient_dict

    def get_mrn_to_patient_id_map(self, mrn_list=None):
        """
        Get a mapping of Medical Record Numbers (MRNs) to patient IDs.

        This method queries the SQL database for all patients with MRNs in the given list
        and returns a dictionary with MRNs as keys and patient IDs as values.

        :param mrn_list: A list of MRNs to filter the patients, or None to get all patients.
        :type mrn_list: list, optional
        :return: A dictionary with MRNs as keys and patient IDs as values.
        :rtype: dict
        """
        if self.metadata_connection_type == "api":
            if not mrn_list:
                return {}

            result_dict = {}
            for mrn in mrn_list:
                result_temp = self._request("GET", f"patients/mrn|{mrn}", params={'time': None})
                result_dict[int(mrn)] = int(result_temp['id'])
            return result_dict

        # If all mrns are in the cache
        if all(int(mrn) in self._mrn_to_patient_id for mrn in mrn_list):
            return {int(mrn): self._mrn_to_patient_id[int(mrn)] for mrn in mrn_list}

        # Refresh the cache and return all available mrns.
        self.get_all_patients()
        return {int(mrn): self._mrn_to_patient_id[int(mrn)] for mrn in mrn_list if int(mrn) in self._mrn_to_patient_id}

    def get_patient_id_to_mrn_map(self, patient_id_list=None):
        """
        Get a mapping of patient IDs to Medical Record Numbers (MRNs).

        This method queries the SQL database for all patients with IDs in the given list
        and returns a dictionary with patient IDs as keys and MRNs as values.

        :param patient_id_list: A list of patient IDs to filter the patients, or None to get all patients.
        :type patient_id_list: list, optional
        :return: A dictionary with patient IDs as keys and MRNs as values.
        :rtype: dict
        """
        # Query the SQL database for all patients with IDs in the given list
        patient_list = self.sql_handler.select_all_patients_in_list(patient_id_list=patient_id_list)

        # Return a dictionary with patient IDs as keys and MRNs as values
        return {row[0]: row[1] for row in patient_list}

    def insert_patient(self, patient_id: int = None, mrn: int = None, gender: str = None, dob: int = None,
                       first_name: str = None, middle_name: str = None, last_name: str = None, first_seen: int = None,
                       last_updated: int = None, source_id: int = 1, weight: float = None, weight_units: str = None,
                       height: float = None, height_units: str = None):
        """
        .. _insert_patient_label:

        Inserts a new patient record into the database with the provided patient details.

        All patient details are optional, but it is recommended to provide as much information as possible
        to ensure accurate patient identification and to avoid duplicate records.

        >>> # Insert a new patient record.
        >>> new_patient_id = sdk.insert_patient(patient_id=123, mrn="123456", gender="M", dob=946684800000000000,
        >>>                                     first_name="John", middle_name="Doe", last_name="Smith",
        >>>                                     first_seen=1609459200000000000, last_updated=1609459200000000000, source_id=1)

        :param int patient_id: A unique number identifying the patient.
        :param str mrn: The Medical Record Number (MRN) of the patient.
        :param str gender: The gender of the patient (e.g., "M", "F", "O" for Other, or "U" for Unknown).
        :param int dob: The date of birth of the patient as a nanosecond epoch.
        :param str first_name: The first name of the patient.
        :param str middle_name: The middle name of the patient.
        :param str last_name: The last name of the patient.
        :param int first_seen: The date when the patient was first seen as a nanosecond epoch.
        :param int last_updated: The date when the patient record was last updated as a nanosecond epoch.
        :param int source_id: The unique identifier of the source from which the patient information was obtained.
        :param float weight: The patients current weight. The time recorded for this weight measurement in the patient
         history table will be the current time. If you want to make it another time use insert_patient_history instead.
        :param str weight_units: The units of the patients weight. This must be specified if inserting a weight.
        :param float height: The patients current height. The time recorded for this height measurement in the patient
         history table will be the current time. If you want to make it another time use insert_patient_history instead.
        :param str height_units: The units of the patients height. This must be specified if inserting a height.

        :return: The unique identifier of the inserted patient record.
        :rtype: int
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        if patient_id is not None:
            patient_info = self.get_patient_info(patient_id)
            if patient_info is not None:
                return patient_id

        if mrn is not None:
            mrn_patient_id = self.get_patient_id(mrn)
            if mrn_patient_id is not None:
                return mrn_patient_id

        # Insert the patient with null for height and weight since it will be updated by
        patient_id = self.sql_handler.insert_patient(patient_id, mrn, gender, dob, first_name, middle_name, last_name,
                                                 first_seen, last_updated, source_id)

        # current time will be the time for the weight and height
        insert_time = time.time_ns()

        # insert the weight into the patient history table. This will update the weight on the patient table
        if weight is not None:
            if weight_units is None:
                raise ValueError("You must specify the units if you are specifying a weight")
            self.insert_patient_history(field='weight', value=weight, units=weight_units, time=insert_time, patient_id=patient_id)

        # insert the height into the patient history table. This will update the height on the patient table
        if height is not None:
            if height_units is None:
                raise ValueError("You must specify the units if you are specifying a height")
            self.insert_patient_history(field='height', value=height, units=height_units, time=insert_time, patient_id=patient_id)

        return patient_id

    def get_patient_history(self, patient_id: int = None, mrn: int = None, field: str = None, start_time: int = None,
                            end_time: int = None, time_units: str = None):
        """
        Retrieve a list of a patients historical measurements using either their numeric patient id or medical record number (MRN).
        If start_time and end_time are left empty it will give all the patient's history. The results are returned in ascending order by time.

        :param int patient_id: The numeric identifier for the patient.
        :param int mrn: The medical record number for the patient.
        :param str field: Which part of the patients history do you want, None will get you all the fields.
            Valid options are 'height', 'weight' or None.
        :param int start_time: The starting epoch time for the range of time you want the patient's history. If none it will get all history before the end_time.
        :param int end_time: The end epoch time for the range of time you want the patient's history. If none it will get all history after the start_time.
        :param str time_units: (Optional) Units for the time. Valid options are 'ns', 's', 'ms', and 'us'. Default is nanoseconds.

        :return: A list of tuples containing the value of the measurement, the units the value is measured in and the
        epoch timestamp of when the measurement was taken. [(3.3, 'kg', 1483264800000000000), (3.4, 'kg', 1483268400000000000)]

        :raises ValueError: If both patient_id and mrn are not provided or neither of them are provided or if start_time is >= end_time or invalid time_unit/field entered.
        """
        # Check if we have either patient ID or MRN
        if patient_id is None and mrn is None:
            raise ValueError("Either patient_id or mrn must be provided.")
        # make sure they supply only one of patient id or mrn
        if patient_id is not None and mrn is not None:
            raise ValueError("Only one of patient_id or mrn should be provided.")
        # check to make sure a proper field was entered
        if field not in ('height', 'weight', None):
            raise ValueError("Invalid field. Expected either 'height' or 'weight'")
        # check that start_time is not greater than end time
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ValueError("Start_time cannot be >= end_time")

        # Handle time units and conversion to nanoseconds
        if time_units:
            if time_units not in time_unit_options.keys():
                raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")
            if start_time is not None:
                start_time *= time_unit_options[time_units]
            if end_time is not None:
                end_time *= time_unit_options[time_units]

        # if the end time is none set it to 10 seconds into the future so you get all data after the start_time
        if end_time is None:
            end_time = time.time_ns() + 10_000_000_000
        # if the start time is none set it to 0 so you get all data before the end_time
        if start_time is None:
            start_time = 0

        # Check if we are in API mode
        if self.metadata_connection_type == "api":
            params = {'field': field, 'start_time': start_time, 'end_time': end_time}
            if patient_id is not None:
                return self._request("GET", f"/patients/id|{patient_id}/history", params=params)
            # if there is no patient_id that means an mrn is used as the identifier
            return self._request("GET", f"/patients/mrn|{mrn}/history", params=params)

        # get the patient id if an mrn was provided
        if mrn is not None:
            patient_id = self.get_patient_id(mrn)

        # if the patient was not found return none
        if patient_id is None:
            return None

        return self.sql_handler.select_patient_history(patient_id, field, start_time, end_time)

    def insert_patient_history(self, field: str, value: float, units: str, time: int, time_units: str = None, patient_id: int = None, mrn: int = None):
        """
        Insert a patient history record using either their numeric patient id or medical record number (MRN).

        :param str field: Which part of the patients history you want to insert. Valid options are 'height' or 'weight'.
        :param float value: The value of the measurement you want to insert.
        :param str units: The units of the measurement you want to insert
        :param int time: The epoch timestamp of the time the measurement was taken.
        :param str time_units: (Optional) Units for the time. Valid options are 'ns', 's', 'ms', and 'us'. Default is nanoseconds.
        :param int patient_id: The numeric identifier for the patient.
        :param int mrn: The medical record number for the patient.

        :return: A list of tuples containing the value of the measurement, the units the value is measured in and the
        epoch timestamp of when the measurement was taken. [(3.3, 'kg', 1483264800000000000), (3.4, 'kg', 1483268400000000000)]

        :raises ValueError: If both patient_id and mrn are not provided or neither of them are provided or if start_time is >= end_time or invalid time_unit/field entered.
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Handle time units and conversion to nanoseconds
        if time_units:
            if time_units not in time_unit_options.keys():
                raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")
            time *= time_unit_options[time_units]

        # Check if we have either patient ID or MRN
        if patient_id is None and mrn is None:
            raise ValueError("Either patient_id or mrn must be provided.")
        # make sure they supply only one of patient id or mrn
        if patient_id is not None and mrn is not None:
            raise ValueError("Only one of patient_id or mrn should be provided.")

        # if they supplied an mrn convert it to a patient_id
        if mrn is not None:
            patient_id = self.get_patient_id(int(mrn))

        return self.sql_handler.insert_patient_history(patient_id, field, value, units, time)

    def get_patient_history_fields(self):
        """
        Returns a list of all strings in the field column of patient history.

        :return: A list of strings of all history fields
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for this method.")

        return self.sql_handler.select_unique_history_fields()

    def get_device_patient_data(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                                mrn_list: List[int] = None, start_time: int = None, end_time: int = None):
        """
        .. _get_device_patient_data_label:

        Retrieves device-patient mappings from the dataset's database based on the provided search criteria.

        The method returns a list of tuples, where each tuple contains four integer values in the following order:
        - device_id (int): The ID of the device associated with the patient.
        - patient_id (int): The ID of the patient associated with the device.
        - start_time (int): The start time as a nanoseconds epoch of the association between the device and the patient.
        - end_time (int): The end time as a nanoseconds epoch of the association between the device and the patient.

        :param List[int] optional device_id_list: A list of device IDs.
        :param List[int] optional patient_id_list: A list of patient IDs.
        :param List[int] optional mrn_list: A list of MRNs (medical record number).
        :param int optional start_time: The start time as a nanoseconds epoch of the device-patient association.
        :param int optional end_time: The end time as a nanoseconds epoch of the device-patient association.
        :return: A list of tuples containing device-patient mapping data, where each tuple contains four integer values in
            the following order: device_id, patient_id, start_time, and end_time.
        :rtype: List[Tuple[int, int, int, int]]

        >>> # Retrieve device-patient mappings from the dataset's database.
        >>> device_id_list = [1, 2]
        >>> patient_id_list = [3, 4]
        >>> start_time = 164708400_000_000_000
        >>> end_time = 1647094800_000_000_000
        >>> device_patient_data = sdk.get_device_patient_data(device_id_list=device_id_list,
        >>>                                                    patient_id_list=patient_id_list,
        >>>                                                    start_time=start_time,
        >>>                                                    end_time=end_time)
        """
        if self.metadata_connection_type == "api":
            return self._api_get_device_patient_data(device_id_list=device_id_list, patient_id_list=patient_id_list,
                                                     mrn_list=mrn_list, start_time=start_time, end_time=end_time)

        if mrn_list is not None:
            patient_id_list = [] if patient_id_list is None else patient_id_list
            mrn_to_patient_id_map = self.get_mrn_to_patient_id_map(mrn_list)
            patient_id_list.extend([mrn_to_patient_id_map[mrn] for mrn in mrn_list if mrn in mrn_to_patient_id_map])

        return self.sql_handler.select_device_patients(
            device_id_list=device_id_list, patient_id_list=patient_id_list, start_time=start_time, end_time=end_time)

    def _api_get_device_patient_data(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                                     mrn_list: List[int] = None, start_time: int = None, end_time: int = None):

        start_time = 0 if start_time is None else start_time
        end_time = time.time_ns() if end_time is None else end_time
        # Determine the list of patient identifiers
        patient_identifiers = []
        if patient_id_list is not None:
            patient_identifiers.extend([f'id|{pid}' for pid in patient_id_list])
        if mrn_list is not None:
            patient_identifiers.extend([f'mrn|{mrn}' for mrn in mrn_list])
        if not patient_identifiers:
            patient_identifiers = [f'id|{pid}' for pid in self.get_all_patients().keys()]

        result = []
        # Query each patient identifier
        for pid in patient_identifiers:
            if pid.split('|')[0] == "id":
                patient_id = int(pid.split('|')[1])
            elif pid.split('|')[0] == "mrn":
                mrn = int(pid.split('|')[1])
                patient_id = self.get_patient_id(mrn)
            else:
                raise ValueError(f"got {pid.split('|')[0]}, expected mrn or id")
            params = {'start_time': start_time, 'end_time': end_time}
            devices_result = self._request("GET", f"patients/{pid}/devices", params=params)

            if devices_result:
                for device in devices_result:
                    query_device_id = device['device_id']
                    query_start_time = device['start_time']
                    query_end_time = device['end_time']
                    # Filter based on device_id_list if it's provided
                    if device_id_list is None or query_device_id in device_id_list:
                        result.append((query_device_id, patient_id, query_start_time, query_end_time))

        return result

    def insert_device_patient_data(self, device_patient_data: List[Tuple[int, int, int, int]]):
        """
        .. _insert_device_patient_data_label:

        Inserts device-patient mappings into the dataset's database.

        The `device_patient_data` parameter is a list of tuples, where each tuple contains four integer values in the
        following order:
        - device_id (int): The ID of the device associated with the patient.
        - patient_id (int): The ID of the patient associated with the device.
        - start_time (int): The start time (in UNIX nano timestamp format) of the association between the device and the patient.
        - end_time (int): The end time (in UNIX nano timestamp format) of the association between the device and the patient.

        The `start_time` and `end_time` values represent the time range in which the device is associated with the patient.

        >>> # Insert a device-patient mapping into the dataset's database.
        >>> device_patient_data = [(1, 2, 1647084000_000_000_000, 1647094800_000_000_000),
        >>>                        (1, 3, 1647084000_000_000_000, 1647094800_000_000_000)]
        >>> sdk.insert_device_patient_data(device_patient_data)

        :param List[Tuple[int, int, int, int]] device_patient_data: A list of tuples containing device-patient mapping
            data, where each tuple contains four integer values in the following order: device_id, patient_id, start_time,
            and end_time.
        :return: None
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Cast all columns to their correct datatype.
        device_patient_data = [(int(device_id), int(patient_id), int(start_time), int(end_time)) for
                               device_id, patient_id, start_time, end_time in device_patient_data]
        self.sql_handler.insert_device_patients(device_patient_data)

    def convert_patient_to_device_id(self, start_time: int, end_time: int, patient_id: int = None, mrn: int = None):
        """
        Converts a patient ID or MRN to a device ID based on the specified time range.

        :param int start_time: Start time for the association.
        :param int end_time: End time for the association.
        :param int patient_id: Patient ID to be converted.
        :param int mrn: MRN to be converted.
        :return: Device ID if a single device fully encapsulates the time range, otherwise None.
        :rtype: int or None
        """

        # Retrieve device-patient mapping data
        if patient_id is not None:
            device_patient_data = self.get_device_patient_data(patient_id_list=[patient_id], start_time=start_time,
                                                               end_time=end_time)
        elif mrn is not None:
            device_patient_data = self.get_device_patient_data(mrn_list=[mrn], start_time=start_time, end_time=end_time)
        else:
            raise ValueError("You must specify either patient_id or mrn.")

        # Group data by device_id
        device_intervals = {}
        for device_id, _, device_start, device_end in device_patient_data:
            if device_id not in device_intervals:
                device_intervals[device_id] = []
            device_intervals[device_id].append([device_start, device_end])

        # Merge overlapping intervals for each device
        for device_id in device_intervals:
            intervals = sorted(device_intervals[device_id], key=lambda x: x[0])
            merged_intervals = [intervals[0]]
            for current in intervals[1:]:
                last = merged_intervals[-1]
                if last[1] >= current[0]:  # Overlapping intervals
                    last[1] = max(last[1], current[1])
                else:
                    merged_intervals.append(current)
            device_intervals[device_id] = merged_intervals

        # Check for a device whose interval encapsulates the provided time range
        matching_devices = []
        for device_id, intervals in device_intervals.items():
            for interval in intervals:
                if interval[0] <= start_time and interval[1] >= end_time:
                    matching_devices.append(device_id)

        # Raise error if more than one match is found
        if len(matching_devices) > 1:
            raise ValueError(f"Multiple devices ({matching_devices}) found for the same time range with parameters: "
                             f"start_time={start_time}, end_time={end_time}, patient_id={patient_id}, mrn={mrn}. "
                             "Please check and fix the device_patient table.")

        return matching_devices[0] if matching_devices else None

    def convert_device_to_patient_id(self, start_time: int, end_time: int, device):
        """
        Converts a device ID or tag to a patient ID based on the specified time range.

        :param int start_time: Start time for the association.
        :param int end_time: End time for the association.
        :param device: Device ID (int) or tag (str) to be converted.
        :return: Patient ID if a single patient's interval encapsulates the time range, otherwise None.
        :rtype: int or None
        """

        # Convert device tag to device ID if necessary
        if isinstance(device, str):
            device_id = self.get_device_id(device)
        elif isinstance(device, int):
            device_id = device
        else:
            raise ValueError(f"device must be either int or str (id or tag), not type{type(device)}")

        # Retrieve device-patient mapping data
        device_patient_data = self.get_device_patient_data(device_id_list=[device_id], start_time=start_time,
                                                           end_time=end_time)

        # Group data by patient_id
        patient_intervals = {}
        for _, patient_id, patient_start, patient_end in device_patient_data:
            if patient_id not in patient_intervals:
                patient_intervals[patient_id] = []
            patient_intervals[patient_id].append([patient_start, patient_end])

        # Merge overlapping intervals for each patient
        for patient_id in patient_intervals:
            intervals = sorted(patient_intervals[patient_id], key=lambda x: x[0])
            merged_intervals = [intervals[0]]
            for current in intervals[1:]:
                last = merged_intervals[-1]
                if last[1] >= current[0]:  # Overlapping intervals
                    last[1] = max(last[1], current[1])
                else:
                    merged_intervals.append(current)
            patient_intervals[patient_id] = merged_intervals

        # Check for a patient whose interval encapsulates the provided time range
        matching_patients = []
        for patient_id, intervals in patient_intervals.items():
            for interval in intervals:
                if interval[0] <= start_time and interval[1] >= end_time:
                    matching_patients.append(patient_id)

        # Raise error if more than one match is found
        if len(matching_patients) > 1:
            raise ValueError(f"Multiple patients ({matching_patients}) found for the same time range with parameters: "
                             f"start_time={start_time}, end_time={end_time}, device={device}. "
                             "Please check and fix the device_patient table.")

        return matching_patients[0] if matching_patients else None

    def get_labels(self, label_name_id_list=None, name_list=None, device_list=None, start_time=None, end_time=None,
                   time_units: str = None, patient_id_list=None,
                   label_source_list: Optional[List[Union[str, int]]] = None, include_descendants=True, limit=None, offset=0):
        """
        Retrieve labels from the database based on specified criteria.

        :param List[int] label_name_id_list: List of label set IDs to filter by.
        :param List[str] name_list: List of label names to filter by. Mutually exclusive with `label_name_id_list`.
        :param List[Union[int, str]] device_list: List of device IDs or device tags to filter by.
        :param int start_time: Start time filter for the labels.
        :param int end_time: End time filter for the labels.
        :param str time_units: Units for the `start_time` and `end_time` filters. Valid options are 'ns', 's', 'ms', and 'us'.
        :param List[int] patient_id_list: List of patient IDs to filter by.
        :param Optional[List[Union[str, int]]] label_source_list: List of label source names or IDs to filter by.
        :param bool include_descendants: Returns all labels of descendant label_name, using requested_name_id and
            requested_name to represent the label name of the requested parent.
        :param int limit: Maximum number of rows to return.
        :param int offset: Offset this number of rows before starting to return labels. Used in combination with limit.

        :return: A list of matching labels from the database. Each label is represented as a dictionary containing label details.
        :rtype: List[Dict]

        Example::

            Given an input filtering by a particular device ID, the output could look like:

            [
                {
                    'label_entry_id': 1,
                    'label_name_id': 10,
                    'label_name': 'example_name_1',
                    'requested_name_id': 10,
                    'requested_name': 'example_name_1',
                    'device_id': 1001,
                    'device_tag': 'tag_1',
                    'patient_id': 12345,
                    'mrn': 7654321,
                    'start_time_n': 1625000000000000000,
                    'end_time_n': 1625100000000000000,
                    'label_source_id': 4,
                    'label_source': "LabelStudio_Project_1",
                },
                ...
            ]

        Note:
            - This method currently only supports database connection mode and not API mode.
            - Either `device_list` or `patient_id_list` should be provided, but not both.
            - Either `label_name_id_list` or `name_list` should be provided, but not both.

        """
        # Ensure either label_name_id_list or name_list is provided, but not both
        if label_name_id_list and name_list:
            raise ValueError("Only one of label_name_id_list or name_list should be provided.")

        # Ensure either device list or patient id list is provided, but not both
        if device_list and patient_id_list:
            raise ValueError("Only one of device_list or patient_id_list should be provided.")

        # Convert time using the provided time units, if specified
        if time_units:
            if time_units not in time_unit_options.keys():
                raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

            if start_time:
                start_time *= time_unit_options[time_units]
            if end_time:
                end_time *= time_unit_options[time_units]

        if self.metadata_connection_type == "api":
            return self._api_get_labels(label_name_id_list, name_list, device_list, start_time, end_time,
                                        patient_id_list, label_source_list, include_descendants, limit, offset)

        # Convert label names to IDs if name_list is used
        if name_list:
            name_id_list = [self.get_label_name_id(name) for name in name_list]
            for label_name, label_id in zip(name_list, name_id_list):
                if label_id is None:
                    raise ValueError(f"Label name '{label_name}' not found in the database.")
            label_name_id_list = name_id_list

        closest_requested_ancestor_dict = {}
        if label_name_id_list and include_descendants:
            label_name_id_list, closest_requested_ancestor_dict = collect_all_descendant_ids(
                label_name_id_list, self.sql_handler)

        # Convert device tags to IDs
        if device_list:
            device_id_list = []
            for device in device_list:
                device_id = self.get_device_id(device) if isinstance(device, str) else device
                if device_id is None:
                    raise ValueError(f"Device Tag {device} not found in database")
                device_id_list.append(device_id)
            device_list = device_id_list

        label_source_id_list = []
        if label_source_list:
            for source in label_source_list:
                if isinstance(source, str):
                    label_source_id = self.get_label_source_id(source)
                    if label_source_id is None:
                        raise ValueError(f"Label source name '{source}' not found in the database.")
                    label_source_id_list.append(label_source_id)
                elif isinstance(source, int):
                    label_source_id_list.append(source)
                else:
                    raise ValueError("Label source list items must be either string (name) or integer (ID).")

        labels = self.sql_handler.select_labels(
            label_set_id_list=label_name_id_list,
            device_id_list=device_list,
            patient_id_list=patient_id_list,
            start_time_n=start_time,
            end_time_n=end_time,
            label_source_id_list=label_source_id_list if label_source_id_list else None,
            limit=limit, offset=offset
        )

        # Extract unique label_set_ids and device_ids
        unique_label_set_ids = {label[1] for label in labels}
        unique_device_ids = {label[2] for label in labels}

        # Create dictionaries for label set and device info for optimized lookup
        label_set_id_to_info = {label_set_id: self.get_label_name_info(label_set_id) for label_set_id in
                                unique_label_set_ids}
        device_id_to_info = {device_id: self.get_device_info(device_id) for device_id in unique_device_ids}

        result = []
        for label_entry_id, label_set_id, device_id, start_time_n, end_time_n, label_source_id in labels:
            requested_id = closest_requested_ancestor_dict.get(label_set_id, label_set_id)
            requested_name = self.get_label_name_info(requested_id)['name']

            # Get label_source_name
            label_source_info = self.get_label_source_info(label_source_id) if label_source_id else None
            label_source_name = label_source_info['name'] if label_source_info else None

            patient_id = self.convert_device_to_patient_id(
                start_time=start_time_n, end_time=end_time_n, device=device_id)
            mrn = None if patient_id is None else self.get_mrn(patient_id)

            formatted_label = {
                'label_entry_id': label_entry_id,
                'label_name_id': label_set_id,
                'label_name': label_set_id_to_info[label_set_id]['name'],
                'requested_name_id': requested_id,
                'requested_name': requested_name,
                'device_id': device_id,
                'device_tag': device_id_to_info[device_id]['tag'],
                'patient_id': patient_id,
                'mrn': mrn,
                'start_time_n': start_time_n,
                'end_time_n': end_time_n,
                'label_source_id': label_source_id,
                'label_source': label_source_name
            }
            result.append(formatted_label)

        return result

    def _api_get_labels(self, label_name_id_list=None, name_list=None, device_list=None, start_time=None, end_time=None,
                        patient_id_list=None, label_source_list: Optional[List[Union[str, int]]] = None,
                        include_descendants=True, limit=None, offset=0):
        limit = 1000 if limit is None else limit

        label_list = []
        while True:
            params = {
                'label_name_id_list': label_name_id_list,
                'name_list': name_list,
                'device_list': device_list,
                'start_time': start_time,
                'end_time': end_time,
                'patient_id_list': patient_id_list,
                'label_source_list': label_source_list,
                'include_descendants': include_descendants,
                'limit': limit,
                'offset': offset
            }

            result_temp = self._request("POST", "labels/", json=params)

            for label in result_temp:
                label_list.append(label)

            # nothing at all was found
            if len(label_list) == 0:
                raise ValueError("No labels found for current search params.")

            # this stops the loop when we have received the last batch of labels from the api
            if len(result_temp) == 0:
                break
            offset += limit

        return label_list

    def insert_label(self, name: str, start_time: int, end_time: int, device: Union[int, str] = None,
                     patient_id: int = None, mrn: int = None, time_units: str = None, label_source: Union[str, int] = None):
        """
        Insert a label record into the database.

        :param str name: Name of the label type.
        :param int start_time: Start time for the label.
        :param int end_time: End time for the label.
        :param Union[int, str] device: Device ID or device tag (exclusive with device and patient_id).
        :param int patient_id: Patient ID for the label to be inserted (exclusive with device and mrn).
        :param int mrn: MRN for the label to be inserted (exclusive with device and patient_id).
        :param str time_units: Units for the `start_time` and `end_time`. Valid options are 'ns', 's', 'ms', and 'us'.
        :param Union[str, int] label_source: Name or ID of the label source.
        :raises ValueError: If the provided label_source is not found in the database.
        :return: The ID of the inserted label

        Example usage:

        .. code-block:: python

            # Insert a label for a device with ID 42
            insert_label(name='Sleep Stage', start_time=1609459200_000_000_000, end_time=1609462800_000_000_000, device=42)

            # Insert a label for a patient with patient_id 12345
            insert_label(name='Medication', start_time=1609459200_000, end_time=1609462800_000, patient_id=12345, time_units='ms')

            # Using a device tag instead of device ID
            insert_label(name='Arrhythmia', start_time=1609459200_000_000_000, end_time=1609462800_000_000_000, device='device-tag-xyz')

            # Specifying time units and label source by name
            insert_label(name='Exercise', start_time=1609459200, end_time=1609462800, device=42, time_units='s', label_source='Manual Entry')

        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Ensure exclusivity of device, patient_id, and mrn
        provided_params = [device is not None, patient_id is not None, mrn is not None]
        if sum(provided_params) > 1:
            raise ValueError("Only one of device, patient_id, or mrn can be provided.")

        # Convert patient_id or mrn to device ID if necessary
        converted_device_id = None
        if patient_id is not None:
            converted_device_id = self.convert_patient_to_device_id(start_time, end_time, patient_id=patient_id)
        elif mrn is not None:
            converted_device_id = self.convert_patient_to_device_id(start_time, end_time, mrn=mrn)

        # Convert device tag to device ID if necessary
        if isinstance(device, str):
            converted_device_id = self.get_device_id(device)
        elif isinstance(device, int):
            converted_device_id = device

        if converted_device_id is None or (isinstance(device, int) and self.get_device_info(device) is None):
            raise ValueError(f"device not found for device {device} patient_id {patient_id} mrn {mrn}")

        # Convert time using the provided time units
        time_units = "ns" if time_units is None else time_units
        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        start_time *= time_unit_options[time_units]
        end_time *= time_unit_options[time_units]

        # Determine label source ID
        if isinstance(label_source, str):
            label_source_id = self.get_label_source_id(label_source)
            if label_source_id is None:
                warnings.warn(f"Label source {label_source} was not found in the database, inserting it now.")
                label_source_id = self.insert_label_source(name=label_source)

        elif isinstance(label_source, int):
            label_source_id = label_source
        else:
            label_source_id = None

        if name not in self._label_set_ids:
            label_id = self.sql_handler.insert_label_set(name)
            self._label_sets[label_id] = {'id': label_id, 'name': name}
            self._label_set_ids[name] = label_id
        else:
            label_id = self._label_set_ids[name]

        # Insert the label into the database
        return self.sql_handler.insert_label(label_id, converted_device_id, start_time, end_time, label_source_id)

    def insert_labels(self, labels: List[Tuple[str, Union[int, str], int, int, Union[str, int, None]]],
                      time_units: str = None, source_type: str = None):
        """
        Insert multiple label records into the database.

        :param List[Tuple[str, Union[int, str], int, int, Union[str, int]]] labels: A list of labels. Each label is a tuple containing:
            - Name of the label type.
            - Source ID based on the source_type parameter (device ID, device tag, patient ID, or MRN).
            - Start time for the label.
            - End time for the label.
            - Name or ID of the label source. (Can be None, for no specified source)
        :param str time_units: Units for the `start_time` and `end_time` of each label. Valid options are 'ns', 's', 'ms', and 'us'. (default ns)
        :param str source_type: The type of source ID provided in the labels. Valid options are 'device_id', 'device_tag', 'patient_id', and 'mrn'.
        :raises ValueError: If the provided label_source or source_type is not found in the database.

        Example usage:

        .. code-block:: python

            # Using device ID as the source type
            labels_data = [
                ('Sleep Stage', 42, 1609459200_000_000_000, 1609462800_000_000_000, None),
                ('Medication', 56, 1609459200_000_000_000, 1609462800_000_000_000, 'Medication DB')
            ]
            insert_labels(labels=labels_data, time_units='s', source_type='device_id')

            # Using MRN as the source type
            labels_data = [
                ('Heart Rate', 1234567, 1609459200_000, 1609462800_000, None),
                ('Blood Pressure', 1234568, 1609459200_000, 1609462800_000, 'Automatic Device')
            ]
            insert_labels(labels=labels_data, time_units='ms', source_type='mrn')

        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        valid_source_types = ["device_id", "device_tag", "patient_id", "mrn"]
        source_type = "device_id" if source_type is None else source_type
        if source_type not in valid_source_types:
            raise ValueError(f"Invalid source type. Expected one of: {', '.join(valid_source_types)}")

        formatted_labels = []

        for label in labels:
            name, source_id, start_time, end_time, label_source = label

            # Convert source_id based on the source_type parameter
            if source_type == "device_tag":
                device = self.get_device_id(source_id)
                if device is None:
                    raise ValueError(f"device tag {source_id} not found in database")
            elif source_type == "patient_id":
                device = self.convert_patient_to_device_id(start_time, end_time, patient_id=source_id)
                if device is None:
                    raise ValueError(f"patient id {source_id} not found in database")
            elif source_type == "mrn":
                device = self.convert_patient_to_device_id(start_time, end_time, mrn=source_id)
                if device is None:
                    raise ValueError(f"mrn {source_id} not found in database")
            elif source_type == "device_id":
                device = source_id
                if self.get_device_info(source_id) is None:
                    raise ValueError(f"device id {source_id} not found in database")
            else:
                raise ValueError(f"Invalid source type. Expected one of: {', '.join(valid_source_types)}")

            # Adjust start and end times using time units
            if time_units:
                if time_units not in time_unit_options.keys():
                    raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")
                start_time *= time_unit_options[time_units]
                end_time *= time_unit_options[time_units]

            # Determine label source ID
            if isinstance(label_source, str):
                label_source_id = self.get_label_source_id(label_source)
                if label_source_id is None:
                    warnings.warn(f"Label source {label_source} was not found in the database, inserting it now.")
                    label_source_id = self.insert_label_source(name=label_source)
            elif isinstance(label_source, int):
                label_source_id = label_source
            else:
                label_source_id = None

            if name not in self._label_set_ids:
                label_id = self.sql_handler.insert_label_set(name)
                self._label_sets[label_id] = {'id': label_id, 'name': name}
                self._label_set_ids[name] = label_id
            else:
                label_id = self._label_set_ids[name]

            # Add to the formatted labels list
            formatted_labels.append((label_id, device, label_source_id, start_time, end_time))

        # Insert the labels into the database
        self.sql_handler.insert_labels(formatted_labels)

    def delete_labels(self, label_id_list=None, label_name_id_list=None, name_list=None, device_list=None,
                      start_time=None, end_time=None, time_units: str = None,
                      patient_id_list=None, label_source_list: Optional[List[Union[str, int]]] = None):
        """
        Delete labels from the database based on specified criteria. If no parameters are passed, the method raises an error for safety.

        :param List[int] label_id_list: List of label IDs to delete. Use '*' to delete all labels.
        :param List[int] label_name_id_list: List of label set IDs to filter labels for deletion.
        :param List[str] name_list: List of label names to filter labels for deletion.
        :param List[Union[int, str]] device_list: List of device IDs or device tags to filter labels for deletion.
        :param int start_time: Start time filter for the labels to delete.
        :param int end_time: End time filter for the labels to delete.
        :param str time_units: Units for the `start_time` and `end_time` filters.
        :param List[int] patient_id_list: List of patient IDs to filter labels for deletion.
        :param Optional[List[Union[str, int]]] label_source_list: List of label source names or IDs to filter labels for deletion.
        :raises ValueError: If no parameters are provided, or if invalid parameters are provided.
        :return: None

        Example usage:
            To delete labels for a specific device ID:
                delete_labels(device_list=[1001])

            To delete all labels:
                delete_labels(label_id_list="*")
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for delete.")

        if "protected_mode" not in self.settings_dict:
            raise ValueError(
                "protected_mode is not found in settings table. protected_mode must be set and equal to False")
        if self.settings_dict["protected_mode"] != "False":
            raise ValueError(
                "Cannot perform delete, protected_mode not set to `False`, change in sql table to allow deletion.")

        if all(param is None for param in
               [label_id_list, label_name_id_list, name_list, device_list, start_time, end_time, patient_id_list,
                label_source_list]):
            raise ValueError(
                "No parameters were provided. For safety, you need to specify at least one parameter. Use label_id_list='*' to delete all labels.")

        if label_id_list == "*":
            all_labels = self.get_labels()
            all_label_ids = [label_info['label_id'] for label_info in all_labels]
            return self.sql_handler.delete_labels(all_label_ids)

        elif label_id_list is not None:
            return self.sql_handler.delete_labels(label_id_list)

        filtered_labels = self.get_labels(label_name_id_list=label_name_id_list, name_list=name_list,
                                          device_list=device_list, start_time=start_time, end_time=end_time,
                                          time_units=time_units, patient_id_list=patient_id_list,
                                          label_source_list=label_source_list)
        filtered_label_ids = [label_info['label_id'] for label_info in filtered_labels]
        return self.sql_handler.delete_labels(filtered_label_ids)

    def get_label_name_id(self, name: str):
        """
        Retrieve the identifier of a label type based on its name.

        :param str name: The name of the label type.
        :return: The identifier of the label type.
        :rtype: int
        """
        if self.metadata_connection_type == "api":
            params = {'label_name_id': None, 'label_name': name}
            return self._request("GET", "/labels/name", params=params)

        # Check if the label name is already in the cached label type IDs dictionary
        if name in self._label_set_ids:
            return self._label_set_ids[name]

        # If the label name is not in the cache, query the database using the SQL handler
        label_id = self.sql_handler.select_label_set_id(name)

        # If the label name is not found in the database, return None
        if label_id is None:
            return None

        # If the label name is found in the database, store the ID in the cache
        self._label_set_ids[name] = label_id
        self._label_sets[label_id] = name  # also update the label types cache
        return label_id

    def get_label_name_info(self, label_name_id: int):
        """
        Retrieve information about a specific label set.

        :param int label_name_id: The identifier of the label set to retrieve information for.
        :return: A dictionary containing information about the label set, including its id and name.
        :rtype: dict

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> label_name_id = 1
        >>> label_name_info = sdk.get_label_name_info(label_name_id)
        >>> print(label_name_info)
        {'id': 1,
         'name': 'Label A1',
         'parent_id': 2,
         'parent_name': 'Label Class A'}

        """
        # Check if metadata is fetched using API and call the appropriate method
        if self.metadata_connection_type == "api":
            params = {'label_name_id': label_name_id, 'label_name': None}
            return self._request("GET", "/labels/name", params=params)

        # If label set info is already cached, return it
        if label_name_id in self._label_sets:
            return self._label_sets[label_name_id]

        # Fetch label set info from the SQL database
        row = self.sql_handler.select_label_set(label_set_id=label_name_id)

        # If label set not found in the database, return None
        if row is None:
            return None

        # Unpack the fetched row into individual variables
        label_name_id, label_set_name, parent_id = row

        parent_name = None
        if parent_id is not None:
            parent_name = self.get_label_name_info(parent_id)

        # Create a dictionary with the label set information
        label_set_info = {
            'id': label_name_id,
            'name': label_set_name,
            'parent_id': parent_id,
            'parent_name': parent_name
        }

        # Cache the label set information for future use
        self._label_sets[label_name_id] = label_set_info

        # Return the label set information dictionary
        return label_set_info

    def get_all_label_names(self, limit=None, offset=0) -> dict:
        """
        Retrieve all distinct label names from the database.
        :param int limit: Maximum number of rows to return.
        :param int offset: Offset this number of rows before starting to return labels. Used in combination with limit.
        :return: A dictionary where keys are label IDs and values are dictionaries containing 'id' and 'name' keys.
        :rtype: dict

        .. note:: Skip and limit are used if there are too many label names to return in one get request.
        """
        if self.metadata_connection_type == "api":
            return self._api_get_all_label_names(limit=limit, offset=offset)

        label_tuple_list = self.sql_handler.select_label_sets(limit=limit, offset=offset)

        label_dict = {}
        for label_info in label_tuple_list:
            label_id, label_name, parent_id = label_info
            parent_name = None
            if parent_id is not None:
                parent_name = self.get_label_name_info(parent_id)['name']
            label_dict[label_id] = {
                'id': label_id,
                'name': label_name,
                'parent_id': parent_id,
                'parent_name': parent_name,
            }
        return label_dict

    def _api_get_all_label_names(self, limit=None, offset=0):
        if limit is None:
            limit = 1000
            label_dict = {}
            while True:
                result_dict = self._request("GET", "labels/names", params={'limit': limit, 'offset': offset})

                if len(result_dict) == 0:
                    break
                label_dict.update(result_dict)
                offset += limit
        else:
            label_dict = self._request("GET", "labels/names", params={'limit': limit, 'offset': offset})

        return {int(k): v for k, v in label_dict.items()}

    def get_label_name_children(self, label_name_id: int = None, name: str = None):
        """
        Retrieve all children of a specific label name. You only need to specify one of label_name_id or name.

        :param int label_name_id: The identifier of the label name.
        :param str name: The name of the label.
        :return: A list of dictionaries, each representing a child label set.
        :rtype: list

        >>> sdk = AtriumSDK()
        >>> children_by_id = sdk.get_label_name_children(label_set_id=1)
        >>> for child in children_by_id:
        ...     print(child)
        ... {'id': 2, 'name': 'Label Set A1', 'parent_id': 1, 'parent_name': 'Label Set A'}
        ... {'id': 3, 'name': 'Label Set A2', 'parent_id': 1, 'parent_name': 'Label Set A'}
        >>> children_by_name = sdk.get_label_name_children(name="Label Set B")
        >>> for child in children_by_name:
        ...     print(child)
        ... {'id': 5, 'name': 'Label Set B1', 'parent_id': 4, 'parent_name': 'Label Set B'}

        """
        if self.metadata_connection_type == "api":
            params = {'label_name_id': label_name_id, 'label_name': name}
            return self._request("GET", "/labels/children", params=params)

        if name:
            label_name_id = self.get_label_name_id(name)

        children = self.sql_handler.select_label_name_children(label_name_id)
        return [self.get_label_name_info(child_id) for child_id, _ in children]

    def get_label_name_parent(self, label_name_id: int = None, name: str = None):
        """
        Retrieve the parent of a specific label name. You only need to specify one of label_name_id or name.

        :param int label_name_id: The identifier of the label name.
        :param str name: The name of the label.

        :return: A dictionary representing the parent label set.
        :rtype: dict

        >>> sdk = AtriumSDK()
        >>> parent_by_id = sdk.get_label_name_parent(label_set_id=2)
        >>> print(parent_by_id)
        ... {'id': 1, 'name': 'Label Set A', 'parent_id': None, 'parent_name': None}
        >>> parent_by_name = sdk.get_label_name_parent(name="Label Set A2")
        >>> print(parent_by_name)
        ... {'id': 1, 'name': 'Label Set A', 'parent_id': None, 'parent_name': None}

        """

        if self.metadata_connection_type == "api":
            params = {'label_name_id': label_name_id, 'label_name': name}
            return self._request("GET", "/labels/parent", params=params)

        if name:
            label_name_id = self.get_label_name_id(name)

        parent_id, _ = self.sql_handler.select_label_name_parent(label_name_id)
        if parent_id:
            return self.get_label_name_info(parent_id)
        else:
            return None

    def get_all_label_name_descendents(self, label_name_id: int = None, name: str = None, max_depth: int = None):
        """
        Retrieve a nested dictionary representing the tree of descendants for a given label name.  You only need to specify one of label_name_id or name.

        :param int label_name_id: The identifier of the label name.
        :param str name: The name of the label.
        :param int max_depth: The maximum depth of the tree to retrieve.

        :return: A nested dictionary of label sets representing the descendants tree.
        :rtype: dict
        """

        if self.metadata_connection_type == "api":
            params = {'label_name_id': label_name_id, 'label_name': name, 'depth': max_depth}
            return self._request("GET", "/labels/descendents", params=params)

        # Determine the label_name_id if only the name is provided
        if name and not label_name_id:
            label_name_id = self.sql_handler.select_label_set_id(name)
            if label_name_id is None:
                raise ValueError(f"No label found with the name {name}")

        # Retrieve all descendants
        descendants = self.sql_handler.select_all_label_name_descendents(label_name_id)
        if not descendants:
            return {}  # No descendants found

        # Constructing the nested dictionary
        return self._build_descendants_tree(label_name_id, descendants, max_depth)

    def _build_descendants_tree(self, root_id, descendants, max_depth, current_depth=0):

        if max_depth is not None and current_depth >= max_depth:
            return {}

        tree = {}
        for descendant in descendants:
            if descendant[2] == root_id:  # parent_id of the descendant is root_id
                child_id = descendant[0]
                tree[descendant[1]] = self._build_descendants_tree(child_id, descendants, max_depth, current_depth + 1)

        return tree

    def insert_label_name(self, name: str, label_name_id=None, parent=None) -> int:
        """
        Insert a label name into the database if it doesn't already exist and return the ID.

        :param str name: The name of the label set to insert.
        :param int label_name_id: (Optional) The desired id of the label name to insert.
        :param int | str parent: (Optional) The parent label in the heirarchical tree diagram. If you use an integer it
            will assume this is the parent label name id and if you use a string it will assume it's the parent label name.
        :return: The ID of the label set.
        :rtype: int
        :raises ValueError: If the label name is empty.

        >>> sdk = AtriumSDK()
        >>> label_name_id = sdk.insert_label_name("Example Label name")
        >>> print(label_name_id)
        1
        """
        if not name:
            raise ValueError("The label name cannot be empty.")

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        parent_id = None
        if isinstance(parent, int):
            if self.get_label_name_info(parent) is None:
                raise ValueError(f"Requested Parent {parent} not found, add it using sdk.insert_label_name")
            parent_id = parent
        elif isinstance(parent, str):
            parent_id = self.get_label_name_id(parent)
            if parent_id is None:
                raise ValueError(f"Requested Parent {parent} not found, add it using sdk.insert_label_name")

        # Check if the label set name is already cached
        existing_label_name_id = self._label_set_ids.get(name)

        # If not cached, insert it into the database and update the cache
        if existing_label_name_id is None:
            label_name_id = self.sql_handler.insert_label_set(name, label_set_id=label_name_id, parent_id=parent_id)
            self._label_sets[label_name_id] = {'id': label_name_id, 'name': name}
            self._label_set_ids[name] = label_name_id
            return label_name_id
        elif label_name_id is not None:
            if label_name_id != existing_label_name_id:
                raise ValueError(f"label name id {label_name_id} not equal to the id existing {existing_label_name_id} "
                                 f"for name {name}")

        # Return the label set ID
        return existing_label_name_id

    def get_label_source_id(self, name: str) -> Optional[int]:
        """
        Gets the label source ID from the name of the label source.
        :param name: The name of the label source to look up.
        :return: The label source ID or None if not found.
        """

        if self.metadata_connection_type == 'api':
            params = {'label_source_id': None, 'label_source_name': name}
            return self._request("GET", "/labels/source", params=params)

        return self.sql_handler.select_label_source_id_by_name(name)

    def get_label_source_info(self, label_source_id: int) -> Optional[dict]:
        """
        Retrieve information about a specific label source by its ID.
        :param label_source_id: The identifier for the label source.
        :return: A dictionary containing information about the label source, or None if not found.
        """
        if self.metadata_connection_type == 'api':
            params = {'label_source_id': label_source_id, 'label_source_name': None}
            return self._request("GET", "/labels/source", params=params)

        return self.sql_handler.select_label_source_info_by_id(label_source_id)

    def insert_label_source(self, name: str, description: str = None) -> int:
        """
        Insert a label source into the database if it doesn't already exist and return its ID.
        :param name: The unique name identifier for the label source.
        :param description: A textual description of the label source.
        :return: The ID of the label source.
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")
        return self.sql_handler.insert_label_source(name, description)

    def get_label_time_series(self, label_name=None, label_name_id=None, device_tag=None,
                              device_id=None, patient_id=None, start_time=None, end_time=None,
                              timestamp_array=None, sample_period=None, time_units: str = None,
                              out: np.ndarray = None, label_source_list: Optional[List[Union[str, int]]] = None):
        """
        Retrieve a time series representation for labels from the database based on specified criteria.

        :param str label_name: Name of the label set to filter by. Mutually exclusive with `label_name_id`.
        :param int label_name_id: ID of the label set to filter by. Mutually exclusive with `label_name`.
        :param str device_tag: Tag of the device to filter by. Mutually exclusive with `device_id`.
        :param int device_id: ID of the device to filter by. Mutually exclusive with `device_tag`.
        :param int patient_id: ID of the patient to filter by.
        :param int start_time: Start time filter for the labels.
        :param int end_time: End time filter for the labels.
        :param np.ndarray timestamp_array: Array of timestamps. If not provided, it's generated using `start_time`, `end_time`, and `sample_period`.
        :param int sample_period: Time period between consecutive timestamps. Required if `timestamp_array` is not provided.
        :param str time_units: Units for the `start_time`, `end_time`, and `sample_period` filters. Valid options are 'ns', 's', 'ms', and 'us'.
        :param np.ndarray out: An optional pre-allocated numpy array to hold the result. The shape must match the expected result shape,
            which is the same as `timestamp_array`. Allowed dtypes are integer types or boolean. If provided,
            the results are written into this array in-place. It should be initialized with zeros.
            Otherwise, a new array is allocated.
        :param Optional[List[Union[str, int]]] label_source_list: List of label source names or IDs to filter by.

        :return: An array representing the presence of a label for each timestamp. If a label is present at a given timestamp, the value is 1, otherwise 0.
        :rtype: np.ndarray

        Example:
            Given a label set name, device tag, start and end times, and a sample period, the output could look like:
            [0, 1, 1, 1, 0, 0, ...]

        .. note::
            - This method currently only supports database connection mode and not API mode.
            - Only one of `label_name` or `label_name_id` should be provided.
            - Only one of `device_tag` or `device_id` should be provided.
            - Either `device_id`/`device_tag` or `patient_id` should be provided, but not combinations of both.
            - If using the `out` parameter, ensure its shape matches the expected result shape, and that it is initialized with zeros.

        Raises:
            ValueError: For various reasons including but not limited to the presence of mutually exclusive arguments,
                        absence of required arguments, or invalid time units.
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this method.")

        # Check for the XOR condition for label_name and label_name_id
        if (label_name is not None) == (label_name_id is not None):
            raise ValueError("Either label_name or label_name_id should be provided, but not both.")

        # Check for the XOR condition for device_tag and device_id
        if (device_tag is not None) and (device_id is not None):
            raise ValueError("Either device_tag or device_id should be provided, but not both.")

        # Check for device_id/device_tag or patient_id
        if sum(x is not None for x in [device_id, device_tag, patient_id]) != 1:
            raise ValueError("Exactly one of device_id, device_tag, or patient_id must be provided.")

        # Convert label_name to label_name_id if it's used
        if label_name:
            label_name_id = self.get_label_name_id(label_name)
            if label_name_id is None:
                raise ValueError(f"Label set name '{label_name}' not found in the database.")

        # Convert device_tag to device_id if it's used
        if device_tag:
            device_id = self.get_device_id(device_tag)
            if device_id is None:
                raise ValueError(f"Device Tag {device_tag} not found in database")

        # Handle time units and conversion to nanoseconds
        if time_units:
            if time_units not in time_unit_options.keys():
                raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")

            if start_time:
                start_time *= time_unit_options[time_units]
            if end_time:
                end_time *= time_unit_options[time_units]
            if sample_period:
                sample_period *= time_unit_options[time_units]

            if timestamp_array is not None:
                timestamp_array = convert_to_nanoseconds(timestamp_array, time_units)

        # If timestamp_array is None, create it using start_time, end_time and sample_period
        if timestamp_array is None:
            if not all([start_time, end_time, sample_period]):
                raise ValueError("If timestamp_array is not provided, start_time, end_time, and sample_period must be "
                                 "set in order to generate a timestamp_array using "
                                 "np.arange(start_time, end_time, sample_period)")
            timestamp_array = np.arange(start_time, end_time, sample_period)

        labels = self.get_labels(label_name_id_list=[label_name_id] if label_name_id is not None else None,
                                 device_list=[device_id] if device_id is not None else None, start_time=start_time,
                                 end_time=end_time, patient_id_list=[patient_id] if patient_id is not None else None,
                                 label_source_list=label_source_list)

        # Create a binary array to indicate presence of a label for each timestamp, if not provided.
        if out is not None:
            allowed_dtypes = [np.bool_] + np.sctypes['int']  # Allowed dtypes: boolean and all integer types

            if out.shape != timestamp_array.shape:
                raise ValueError(
                    f"The 'out' array shape {out.shape} doesn't match expected shape {timestamp_array.shape}.")

            if out.dtype not in allowed_dtypes:
                valid_dtypes_str = ", ".join([dtype.__name__ for dtype in allowed_dtypes])
                raise ValueError(f"The 'out' array dtype is {out.dtype}, but expected one of: {valid_dtypes_str}.")

            if not np.all(out == 0):  # Ensure that the out array starts with all zeros
                raise ValueError("The 'out' array should be initialized with zeros. It contains non-zero values.")

            result_array = out
        else:
            result_array = np.zeros(timestamp_array.shape, dtype=np.int8)

        for label in labels:
            start_idx = np.searchsorted(timestamp_array, label['start_time_n'], side='left')
            end_idx = np.searchsorted(timestamp_array, label['end_time_n'], side='right')
            result_array[start_idx:end_idx] = 1

        return result_array

    def get_iterator(self, definition, window_duration, window_slide, gap_tolerance=None,
                     num_windows_prefetch=None, time_units: str = None, label_threshold=0.5,
                     iterator_type=None, window_filter_fn=None, shuffle=False,
                     cached_windows_per_source=None, patient_history_fields=None) -> DatasetIterator:
        """
        Constructs and returns a `DatasetIterator` object that allows iteration over the dataset according to
        the specified definition.

        The method first verifies the provided definition against the dataset of the calling class object.
        If certain parts of the cohort definition aren't present within the dataset, the method will truncate the
        requested cohort to fit the dataset and issue warnings about the dropped data.

        When using a Pytorch DataLoader, ensure that `get_iterator`'s `num_windows_prefetch` is greater than the DataLoader
        `batch_size` * `num_workers` * `prefetch_factor`. If you do this, then the Dataloader will correctly cooperate
        with the Iterator's cache functionality.

        Additionally, when shuffling we recommend you set `cached_windows_per_source = num_windows_prefetch // 100`
        for balanced randomness vs performance.

        :param definition: A DefinitionYAML object or string representation specifying the measures and
                           patients or devices over particular time intervals.
        :type definition: Union[DatasetDefinition, str]
        :param window_duration: Duration of each window in units time_units (default nanoseconds).
        :type window_duration: int
        :param window_slide: Slide duration between consecutive windows in units time_units (default nanoseconds).
        :type window_slide: int
        :param gap_tolerance: Tolerance for gaps in definition intervals auto generated by "all" (optional) in units
            time_units (default nanoseconds)..
        :type gap_tolerance: int, optional
        :param num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
        :type num_windows_prefetch: int, optional
        :param time_units: If you would like the window_duration and window_slide to be specified in units other than
                            nanoseconds you can choose from one of ["s", "ms", "us", "ns"].
        :type time_units: str
        :param label_threshold: The percentage of the window that must contain a label before the entire window is
            marked by that label (eg. 0.5 = 50%). All labels meeting the threshold will be marked.
        :type label_threshold: float
        :param iterator_type: Specify the type of iterator. If set to 'random_access', a RandomAccessDatasetIterator
          will be returned, allowing indexed access to dataset windows. If set to 'filtered',
          a FilteredDatasetIterator will be returned with additional filtering functionality based on
          the `window_filter_fn`. By default or if set to None, a standard DatasetIterator is returned.
        :type iterator_type: str, optional
        :param window_filter_fn: If provided, only windows for which this function returns True will be included in the
             iteration. This function should accept a window as its argument. This is only applicable
             if `iterator_type` is set to 'filtered'.
        :type window_filter_fn: callable, optional
        :param shuffle: If True, the order of windows will be randomized before iteration. If set to an integer, this
            value will seed the random number generator for reproducible shuffling. If False, windows are
            returned in their original order.
        :type shuffle: Union[bool, int], optional
        :param cached_windows_per_source: The maximum number of windows to cache for a single source before moving
            on to a new source, helpful for adding more randomness to the shuffle. Making it too small heavily decreases
            efficiency, making it too large will make the windows less random when shuffled.
        :type cached_windows_per_source: int, optional
        :param list patient_history_fields: A list of patient_info fields you would like returned in the Window object.

        :return: DatasetIterator object to easily iterate over the specified data.
        :rtype: DatasetIterator

        **Example**:

        .. code-block:: python

            sdk = AtriumSDK(dataset_location=local_dataset_location)

            # Define Measures
            measures = ["MLII"]

            # Define Patients and Time Regions
            patient_ids = {
                1: "all",
                2: [{"time0": 1682739250000000000, "pre": 500000000, "post": 500000000}],
                3: [{"start": 1690776318966000000, "end": 1690777625288000000}],
                4: [{"start": 1690781225288000000}],
                5: [{"end": 1690787437932000000}],
            }

            # Create Definition Object
            definition = DefinitionYAML(measures=measures, patient_ids=patient_ids)

            # Get the Iterator Object
            slide_size_nano = window_size_nano = 60_000_000_000  # 1 minute nano
            iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano)

            # Loop over all windows (numpy.ndarray)
            for window in iterator:
                print(window)

        """
        # check that a correct unit type was entered
        time_units = "ns" if time_units is None else time_units
        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        # convert to nanoseconds
        window_duration = int(window_duration * time_unit_options[time_units])
        window_slide = int(window_slide * time_unit_options[time_units])
        if gap_tolerance is not None:
            gap_tolerance = int(gap_tolerance * time_unit_options[time_units])

        max_cache_duration_per_source = None
        if cached_windows_per_source is not None:
            assert isinstance(cached_windows_per_source, int), "cached_windows_per_source must be of type int."
            assert cached_windows_per_source > 0, "cached_windows_per_source must be at least 1."
            max_cache_duration_per_source = window_duration + (window_slide * (cached_windows_per_source - 1))

        validated_measure_list, validated_label_set_list, validated_sources = verify_definition(
            definition, self, gap_tolerance=gap_tolerance)

        # Create appropriate iterator object based on iterator_type
        if iterator_type == 'random_access':
            iterator = RandomAccessDatasetIterator(
                self, validated_measure_list, validated_label_set_list, validated_sources,
                window_duration, window_slide, num_windows_prefetch=num_windows_prefetch,
                label_threshold=label_threshold, time_units=time_units, max_cache_duration=max_cache_duration_per_source,
                shuffle=shuffle, patient_history_fields=patient_history_fields)
        elif iterator_type == 'filtered':
            if window_filter_fn is None:
                raise ValueError("window_filter_fn must be provided when iterator_type is 'filtered'")
            iterator = FilteredDatasetIterator(
                self, validated_measure_list, validated_label_set_list, validated_sources,
                window_duration, window_slide, num_windows_prefetch=num_windows_prefetch,
                label_threshold=label_threshold, time_units=time_units, window_filter_fn=window_filter_fn,
                max_cache_duration=max_cache_duration_per_source, shuffle=shuffle, patient_history_fields=patient_history_fields)
        else:
            iterator = DatasetIterator(
                self, validated_measure_list, validated_label_set_list, validated_sources,
                window_duration, window_slide, num_windows_prefetch=num_windows_prefetch,
                label_threshold=label_threshold, time_units=time_units, shuffle=shuffle,
                max_cache_duration=max_cache_duration_per_source, patient_history_fields=patient_history_fields)

        return iterator

    def get_interval_array(self, measure_id=None, device_id=None, patient_id=None,
                           gap_tolerance_nano: int = 0, start=None, end=None, measure_tag=None,
                           freq=None, units=None, freq_units=None, device_tag=None, mrn=None):
        """
        .. _get_interval_array_label:

        Returns a 2D array representing the availability of a specified measure (signal) and a specified source
        (device id or patient id). Each row of the 2D array output represents a continuous interval of available
        data while the first and second columns represent the start epoch and end epoch of that interval
        respectively.

        >>> measure_id = 21
        >>> device_id = 25
        >>> start_epoch_s = 1669668855
        >>> end_epoch_s = start_epoch_s + 3600  # 1 hour after start.
        >>> start_epoch_nano = start_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
        >>> end_epoch_nano = end_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
        >>> interval_arr = sdk.get_interval_array(measure_id=measure_id, device_id=device_id, start=start_epoch_nano, end=end_epoch_nano)
        >>> interval_arr
        array([[1669668855000000000, 1669668856000000000],
        [1669668857000000000, 1669668858000000000],
        [1669668859000000000, 1669668860000000000],
        [1669668861000000000, 1669668862000000000],
        [1669668863000000000, 1669668864000000000]], dtype=int64)

        :param int measure_id: The measure identifier corresponding to the measures table in the
            linked relational database.
        :param int device_id: The device identifier corresponding to the devices table in the
            linked relational database.
        :param int patient_id: The patient identifier corresponding to the encounter table in the
            linked relational database.
        :param int gap_tolerance_nano: The maximum allowable gap size in the data such that the output considers a
            region continuous. Put another way, the minimum gap size, such that the output of this method will add
            a new row.
        :param int start: The minimum time epoch for which to include intervals.
        :param int end: The maximum time epoch for which to include intervals.
        :param str measure_tag: A short string identifying the signal. Required if measure_id is None.
        :param freq: The sample frequency of the signal. Helpful with measure_tag.
        :param str units: The units of the signal. Helpful with measure_tag.
        :param str freq_units: Units for frequency. Options: ["nHz", "uHz", "mHz",
            "Hz", "kHz", "MHz"] default "nHz".
        :param str device_tag: A string identifying the device. Exclusive with device_id.
        :param int mrn: Medical record number for the patient. Exclusive with patient_id.
        :rtype: numpy.ndarray
        :returns: A 2D array representing the availability of a specified measure.

        """

        if device_id is None and device_tag is not None:
            device_id = self.get_device_id(device_tag)

        if patient_id is None and mrn is not None:
            patient_id = self.get_patient_id(mrn)

        # Check if the metadata connection type is API
        if self.metadata_connection_type == "api":
            if measure_id is None:
                assert measure_tag is not None and freq is not None and units is not None, \
                    "Must provide measure_id or all of measure_tag, freq, units"
                measure_id = self.get_measure_id(measure_tag, freq, units, freq_units)

            params = {'measure_id': measure_id, 'device_id': device_id, 'patient_id': patient_id, 'start_time': start,
                      'end_time': end, 'gap_tolerance': gap_tolerance_nano}
            return self._request("GET", "intervals", params=params)

        # Check the measure
        if measure_id is None:
            assert measure_tag is not None, "One of measure_id, measure_tag must be specified."
            measure_id = get_best_measure_id(self, measure_tag, freq, units, freq_units)

        # Query the database for intervals based on the given parameters
        interval_result = self.sql_handler.select_intervals(
            measure_id, start_time_n=start, end_time_n=end, device_id=device_id, patient_id=patient_id)

        # Initialize an empty list to store the final intervals
        arr = []
        # Iterate through the sorted interval results
        for row in interval_result:
            # If the final intervals list is not empty and the difference between the current interval's start time
            # and the previous interval's end time is less than or equal to the gap tolerance, update the end time
            # of the previous interval
            if arr and row[3] - arr[-1][-1] <= gap_tolerance_nano:
                arr[-1][-1] = row[4]
            # Otherwise, add a new interval to the final intervals list
            else:
                arr.append([row[3], row[4]])

        # Convert the final intervals list to a numpy array with int64 data type
        return np.array(arr, dtype=np.int64)

    def get_bed_id(self, bed_name: str) -> int | None:
        """
        Get the ID for a given bed name.

        :param str bed_name: The name of the bed.
        :return: The bed_id if found, else returns None.
        :rtype: int | None
        """
        bed_data = self.sql_handler.select_bed(name=bed_name)
        if bed_data:
            return bed_data[0]
        return None

    def get_bed_info(self, bed_id: int) -> dict | None:
        """
        Get a dictionary representing the bed information for a given bed ID.

        :param int bed_id: The ID of the bed.
        :return: A dictionary with the bed's information if found, else None.
        :rtype: dict | None
        """
        bed_data = self.sql_handler.select_bed(bed_id=bed_id)
        if bed_data:
            return {"id": bed_data[0], "unit_id": bed_data[1], "name": bed_data[2]}
        return None

    def get_source_id(self, source_name: str) -> int | None:
        """
        Get the ID for a given source name. These are sources of data e.g. Atriumdb, EPIC ect.

        :param str source_name: The name of the source.
        :return: The source_id, None if not found.
        :rtype: int | None
        """
        source_data = self.sql_handler.select_source(name=source_name)
        if source_data:
            return source_data[0]
        return None

    def get_source_info(self, source_id: int) -> dict | None:
        """
        Get a dictionary representing the source information for a given source ID.

        :param int source_id: The ID of the source.
        :return: A dictionary with the source's information if found, else None.
        :rtype: dict | None
        """
        source_data = self.sql_handler.select_source(source_id=source_id)
        if source_data:
            return {"id": source_data[0], "name": source_data[1], "description": source_data[2]}
        return None

    def _request(self, method: str, endpoint: str, **kwargs):

        # Construct the full URL by combining the base API URL and the endpoint.
        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # check if the api token will expire within 30 seconds and if so refresh it
        if self.validate_token and time.time() >= self.token_expiry - 30:
            # get new API token
            self._refresh_token()

        # Set the authorization header using the stored access token.
        headers = {'Authorization': f"Bearer {self.token}"}

        # Send the API request using the specified method, URL, headers, and any additional arguments.
        response = requests.request(method, url, headers=headers, **kwargs)

        # Check if the response has a 200 status code. If not, raise an error.
        if response.status_code != 200:
            raise ValueError(
                f"API request failed with status code {response.status_code}: {response.text} \n url: {url}")

        # Return the JSON response from the API request.
        return response.json()

    def _websocket_connect(self):
        def conn():
            self.websock_conn = connect(f"{self.ws_url}/sdk/blocks/ws", compression=None, max_size=None,
                                        additional_headers={"Authorization": "Bearer {}".format(self.token)})

        # The websockets lib uses a thread to receive messages. Normally you would call close() after receiving the
        # messages to shut that thread down but since we are reducing overhead we want to keep that connection open
        # for the life of the sdk object. If we don't shut it down then the program using the sdk will hang even
        # after the main is finished. Since we have no control over the users or the websocket libs code we have to
        # do this automatically. The only way to do that is to make the receiving thread a daemon thread and since
        # we cant control its creation to set daemon=True, the only way to do that if by making a thread of our own
        # which is a daemon then any threads spawned by that thread will also automatically be a daemon thread.
        # This is why we do the websocket connection in our own thread.
        websocket_connect_thread = threading.Thread(target=conn, daemon=True)
        websocket_connect_thread.start()

        # wait for thread to make the websocket connection
        websocket_connect_thread.join()

    def _refresh_token(self):
        if self.websock_conn is not None:
            # close old websocket connection
            self.websock_conn.close()
            self.websock_conn = None

        # send request to Auth0 to refresh your API token using your refresh token
        token_payload = {'grant_type': 'refresh_token', 'client_id': self.auth_config['auth0_client_id'], 'refresh_token': self.refresh_token}
        token_response = requests.post(f'https://{self.auth_config["auth0_tenant"]}/oauth/token', data=token_payload)

        # parse the response
        token_data = token_response.json()

        if token_response.status_code != 200:
            raise RuntimeError(f"Something went wrong when refreshing your API token. HTTP Error {token_response.status_code}, {token_data}")

        # save new access token
        self.token = token_data['access_token']

        # validate bearer token and get its expiry
        decoded_token = _validate_bearer_token(self.token, self.auth_config)
        self.token_expiry = decoded_token['exp']

        # if the user is using a .env file to store the token
        if self.dot_env_loaded:
            # change the api token in the .env file
            set_key("./.env", "ATRIUMDB_API_TOKEN", token_data['access_token'])
            # load the new environment variables into the OS
            load_dotenv(dotenv_path="./.env", override=True)

        _LOGGER.debug("Expired token refreshed")

    def close(self):
        """
        Close all connections to mariadb or the api. This should be run at the end of your program after you are done
        with the sdk object.
        """

        # make sure we are in api mode and if we are close the connection
        if self.mode == "api" and self.websock_conn is not None:
            self.websock_conn.close()
            _LOGGER.debug("Websocket connection closed")
        # if we are in sql mode and there is a connection pool close it
        elif (self.metadata_connection_type == "mariadb" or self.metadata_connection_type == "mysql") and self.sql_handler.pool is not None:
            self.sql_handler.pool.close()

    def get_filename_dict(self, file_id_list):
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        result_dict = {}

        # Query file index table for file_id, filename pairs
        for row in self.sql_handler.select_files(file_id_list):
            # Add them to a dictionary {file_id: filename}
            result_dict[row[0]] = row[1]

        return result_dict

    def _get_all_settings(self):
        settings = self.sql_handler.select_all_settings()
        return {setting[0]: setting[1] for setting in settings}

    def _overwrite_delete_data(self, measure_id, device_id, new_time_data, time_0, raw_time_type, values_size,
                               freq_nhz):
        # Make assumption for analog
        analog = False

        # Calculate the period in nanoseconds
        period_ns = int((10 ** 18) // freq_nhz)

        # Check if the input data is already a timestamp array
        if raw_time_type == 1:
            end_time_ns = int(new_time_data[-1])
        # Check if the input data is a gap array, and convert it to a timestamp array
        elif raw_time_type == 2:
            end_time_ns = time_0 + (values_size * period_ns) + np.sum(new_time_data[1::2])
            new_time_data = np.arange(time_0, end_time_ns, period_ns, dtype=np.int64)
        else:
            raise ValueError("Overwrite only supported for gap arrays and timestamp arrays.")

        # Initialize dictionary to store overwritten files
        overwrite_file_dict = {}
        # Initialize list to store all old file blocks (The blocks before this latest write_data command)
        all_old_file_blocks = []

        # Get the list of old blocks
        old_block_list = self.sql_handler.select_blocks(measure_id, int(time_0), end_time_ns, device_id)

        # Get the dictionary of old file IDs and their corresponding filenames
        old_file_id_dict = self.get_filename_dict(list(set([row[3] for row in old_block_list])))

        # Iterate through the old files
        for file_id, filename in old_file_id_dict.items():
            # Get the list of blocks for the current file
            file_block_list = self.sql_handler.select_blocks_from_file(file_id)
            # Extend the list of all old file blocks with the current file's blocks
            all_old_file_blocks.extend(file_block_list)

            # Condense the byte read list
            read_list = condense_byte_read_list(file_block_list)

            # Read the data from the old files
            encoded_bytes = self.file_api.read_file_list(read_list, old_file_id_dict)

            # Get the number of bytes for each block
            num_bytes_list = [row[5] for row in file_block_list]

            # Get the start and end times for the current file
            start_time_n = file_block_list[0][6]
            end_time_n = file_block_list[-1][7] * 2  # Just don't truncate output

            # Decode the data from the old files
            old_times, old_values, old_headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                                          time_type=1)
            old_times, old_values = sort_data(old_times, old_values, old_headers, start_time_n, end_time_n,
                                              allow_duplicates=False)
            # Convert old times to int64
            old_times = old_times.astype(np.int64)

            # Get the mask for the difference between old and new times
            diff_mask = np.in1d(old_times, new_time_data, assume_unique=False, invert=True)

            # If there is any difference, process it
            if np.any(diff_mask):
                diff_times, diff_values = old_times[diff_mask], old_values[diff_mask]

                # Get the scale factors and data types from the headers
                freq_nhz = old_headers[0].freq_nhz
                scale_m = old_headers[0].scale_m
                scale_b = old_headers[0].scale_b
                raw_value_type = old_headers[0].v_raw_type
                encoded_value_type = old_headers[0].v_encoded_type

                # Set the raw and encoded time types
                raw_time_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
                encoded_time_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

                # Encode the difference data
                encoded_bytes, encode_headers, byte_start_array = self.block.encode_blocks(
                    diff_times, diff_values, freq_nhz, diff_times[0],
                    raw_time_type=raw_time_type,
                    raw_value_type=raw_value_type,
                    encoded_time_type=encoded_time_type,
                    encoded_value_type=encoded_value_type,
                    scale_m=scale_m,
                    scale_b=scale_b)

                # Write the encoded difference data to a new file
                diff_filename = self.file_api.write_bytes(measure_id, device_id, encoded_bytes)

                # Get the block and interval data for the encoded difference data
                block_data, interval_data = get_block_and_interval_data(
                    measure_id, device_id, encode_headers, byte_start_array, [])

                # Update the overwrite_file_dict with the new file and its associated data
                overwrite_file_dict[diff_filename] = (block_data, interval_data)

            # Return the dictionary of overwritten files, the list of old block IDs, and the list of old file ID pairs
        return overwrite_file_dict, [row[0] for row in old_block_list], list(old_file_id_dict.items())

    def get_data_from_tsc_file(self, file_path, analog=True, time_type=1, sort=True, allow_duplicates=True):
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        encoded_bytes = self.file_api.read_from_filepath(file_path)

        r_times, r_values, headers = self.block.decode_block_from_bytes_alone(
            encoded_bytes, analog=analog, time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            r_times, r_values = sort_data(r_times, r_values, headers, 0, (2**63) - 1, allow_duplicates)

        return headers, r_times, r_values

    def get_batched_data_generator(self, measure_id: int, start_time_n: int = None, end_time_n: int = None,
                                   device_id: int = None, patient_id=None, time_type=1, analog=True,
                                   block_info=None,
                                   max_kbyte_in_memory=None, window_size=None, step_size=None,
                                   get_last_window=True):

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        # Set the step size to the window size if it is not provided
        if window_size is not None and step_size is None:
            step_size = window_size

        # If block_info is not provided, get the block_list and filename_dict
        if block_info is None:
            block_list = self.sql_handler.select_blocks(int(measure_id), start_time_n, end_time_n, device_id, patient_id)

            file_id_list = list(set([row['file_id'] for row in block_list]))
            filename_dict = self.get_filename_dict(file_id_list)
        else:
            block_list = block_info['block_list']
            filename_dict = block_info['filename_dict']

        # Return nothing if there are no blocks
        if len(block_list) == 0:
            return

        # Initialize variables for batch generation
        current_memory_kb = 0
        cur_values = 0
        current_index = 0
        current_blocks_meta = []
        remaining_values = sum([block_metadata['num_values'] for block_metadata in block_list])

        # Iterate through the blocks
        for block_metadata in block_list:
            current_blocks_meta.append(block_metadata)
            current_memory_kb += (block_metadata['num_bytes'] +
                                  (block_metadata['num_values'] * 16)) / 1000
            cur_values += block_metadata['num_values']
            remaining_values -= block_metadata['num_values']

            # Process blocks when memory limit is reached or when enough values are collected for a window
            if current_memory_kb >= max_kbyte_in_memory and (window_size is None or cur_values >= window_size):
                headers, r_times, r_values = self.get_blocks(current_blocks_meta, filename_dict, measure_id,
                                                             start_time_n, end_time_n, analog, time_type)

                yield from yield_data(r_times, r_values, window_size, step_size, get_last_window and
                                      (current_blocks_meta[-1] is block_list[-1]), current_index)

                # Update the current index by adding the size of the current batch of values
                current_index += r_values.size

                # If a window size is specified, calculate the next step
                if window_size is not None:
                    next_step = (((r_values.size - window_size) // step_size) + 1) * step_size
                    times_before, values_before = r_times[next_step:], r_values[next_step:]
                    # Update the current index by subtracting the size of the values before the next step
                    current_index -= values_before.size

                # Clean up memory by removing headers, r_times and r_values
                del headers, r_times, r_values

                # Reset memory usage, current values, and current blocks metadata
                current_memory_kb = 0
                cur_values = 0
                current_blocks_meta = []

        # Process the remaining blocks if there is any memory left
        if current_memory_kb > 0:
            headers, r_times, r_values = self.get_blocks(current_blocks_meta, filename_dict, measure_id,
                                                         start_time_n,
                                                         end_time_n, analog, time_type)

            # If the window size is specified and the size of the current batch of values is smaller than the window size
            if window_size is not None and r_values.size < window_size:
                if get_last_window:
                    current_index += r_values.size
                    last_num_values = 0
                    last_blocks_meta = [block_list[-1]]
                    # Iterate through the blocks in reverse order to collect enough values for the last window
                    for block_metadata in reversed(block_list[:-1]):
                        last_blocks_meta = [block_metadata] + last_blocks_meta
                        last_num_values += block_metadata['num_values']
                        if last_num_values >= window_size:
                            break

                    # Retrieve the last window's data
                    headers, r_times, r_values = self.get_blocks(last_blocks_meta, filename_dict, measure_id,
                                                                 start_time_n, end_time_n, analog, time_type)

                    # Get the last window's data by slicing the time and value arrays
                    r_times, r_values = r_times[-window_size:], r_values[-window_size:]
                    current_index -= window_size

                    # Yield the last window's data if its size matches the window size
                    if r_values.size == window_size:
                        yield from yield_data(r_times, r_values, window_size, step_size, False, current_index)
            else:
                # Yield the current batch's data if the window size condition is not met
                yield from yield_data(r_times, r_values, window_size, step_size, get_last_window, current_index)

    def get_blocks(self, current_blocks_meta, filename_dict, measure_id, start_time_n, end_time_n, analog, time_type=1,
                   sort=True, allow_duplicates=True):

        # Condense the byte read list from the current blocks metadata
        read_list = condense_byte_read_list(current_blocks_meta)

        # Read the data from the files using the measure ID and the read list
        encoded_bytes = self.file_api.read_file_list(read_list, filename_dict)

        # Extract the number of bytes for each block in the current blocks metadata
        num_bytes_list = [row[5] for row in current_blocks_meta]

        # Decode the block array and get the headers, times, and values
        r_times, r_values, headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                              time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates)

        return headers, r_times, r_values

    def write_data_file_only(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray,
                             freq_nhz: int, time_0: int, raw_time_type: int = None, raw_value_type: int = None,
                             encoded_time_type: int = None, encoded_value_type: int = None, scale_m: float = None,
                             scale_b: float = None):

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Block Encode
        encoded_bytes, encode_headers, byte_start_array = self.block.encode_blocks(
            time_data, value_data, freq_nhz, time_0,
            raw_time_type=raw_time_type,
            raw_value_type=raw_value_type,
            encoded_time_type=encoded_time_type,
            encoded_value_type=encoded_value_type,
            scale_m=scale_m,
            scale_b=scale_b)

        # Write to Disk
        filename = self.file_api.write_bytes(measure_id, device_id, encoded_bytes)

        # Calculate Intervals
        intervals = find_intervals(freq_nhz, raw_time_type, time_data, time_0, int(value_data.size))

        encode_headers = [BlockMetadataWrapper(head) for head in encode_headers]

        return measure_id, device_id, filename, encode_headers, byte_start_array, intervals

    def metadata_insert_sql(self, measure_id: int, device_id: int, path: str, metadata: list, start_bytes: np.ndarray,
                            intervals: list):
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        # Get the needed block and interval data from the metadata
        block_data, interval_data = get_block_and_interval_data(
            measure_id, device_id, metadata, start_bytes, intervals)

        # Insert the block and interval data into the metadata table
        self.sql_handler.insert_tsc_file_data(path, block_data, interval_data, None)