import numpy as np

from atriumdb.adb_functions import get_block_and_interval_data, condense_byte_read_list, find_intervals, \
    merge_interval_lists, sort_data, yield_data, convert_to_nanoseconds, convert_to_nanohz, convert_from_nanohz
from atriumdb.block import Block, convert_gap_array_to_intervals, \
    convert_intervals_to_gap_array
from atriumdb.block_wrapper import T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, BlockMetadataWrapper
from atriumdb.file_api import AtriumFileHandler
from atriumdb.helpers import shared_lib_filename_windows, shared_lib_filename_linux, protected_mode_default_setting, \
    overwrite_default_setting
from atriumdb.helpers.block_calculations import calc_time_by_freq, freq_nhz_to_period_ns
from atriumdb.helpers.block_constants import TIME_TYPES
from atriumdb.helpers.settings import ALLOWABLE_OVERWRITE_SETTINGS, PROTECTED_MODE_SETTING_NAME, OVERWRITE_SETTING_NAME, \
    ALLOWABLE_PROTECTED_MODE_SETTINGS
from atriumdb.intervals.intervals import Intervals
from concurrent.futures import ThreadPoolExecutor
import time
import bisect
import requests
import random
from requests import Session
from pathlib import Path, PurePath
from multiprocessing import cpu_count
import sys
from typing import Union, List, Tuple
from functools import cache

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sql_constants import SUPPORTED_DB_TYPES
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import logging

_LOGGER = logging.getLogger(__name__)

DEFAULT_META_CONNECTION_TYPE = 'sqlite'


# _LOGGER.basicConfig(
#     level=_LOGGER.debug,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         _LOGGER.StreamHandler()
#     ]
# )


class AtriumSDK:
    """
    The Core SDK Object that represents a single dataset and provides methods to interact with it.

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
    >>> sdk = AtriumSDK(dataset_location="./example_dataset",
    >>>                 metadata_connection_type=metadata_connection_type,
    >>>                 connection_params=connection_params)

    >>> # Remote API Mode
    >>> api_url = "http://example.com/api/v1"
    >>> token = "4e78a93749ead7893"
    >>> sdk = AtriumSDK(api_url=api_url, token=token)

    :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the dataset will be written.
    :param str metadata_connection_type: Specifies the type of connection to use for metadata. Options are "sqlite", "mysql", "mariadb", or "api". Default "sqlite".
    :param dict connection_params: A dictionary containing connection parameters for "mysql" or "mariadb" connection type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.
    :param int num_threads: Specifies the number of threads to use when processing data.
    :param str api_url: Specifies the URL of the server hosting the API in "api" connection type.
    :param str token: An authentication token for the API in "api" connection type.
    :param str tsc_file_location: A file path pointing to the directory in which the TSC (time series compression) files are written for this dataset. Used to customize the TSC directory location, rather than using `dataset_location/tsc`.
    :param str atriumdb_lib_path: Legacy variable supporting old versions, do not use. A file path pointing to the CDLL that powers the compression and decompression.
    """

    def __init__(self, dataset_location: Union[str, PurePath] = None, metadata_connection_type: str = None,
                 connection_params: dict = None, num_threads: int = None, api_url: str = None, token: str = None,
                 tsc_file_location: str = None, atriumdb_lib_path: str = None):

        metadata_connection_type = DEFAULT_META_CONNECTION_TYPE if \
            metadata_connection_type is None else metadata_connection_type

        assert dataset_location is not None or tsc_file_location is not None
        if dataset_location is None and tsc_file_location is None:
            raise ValueError("dataset_location or tsc_file_location must be specified.")

        if isinstance(dataset_location, str):
            dataset_location = Path(dataset_location)
        if tsc_file_location is None:
            tsc_file_location = dataset_location / 'tsc'

        if num_threads is None:
            num_threads = max(cpu_count() - 2, 1)

        if atriumdb_lib_path is None:

            if sys.platform == "win32":
                shared_lib_filename = shared_lib_filename_windows
            else:
                shared_lib_filename = shared_lib_filename_linux

            this_file_path = Path(__file__)
            atriumdb_lib_path = this_file_path.parent.parent / shared_lib_filename

        self.block = Block(atriumdb_lib_path, num_threads)
        self.sql_handler = None

        if metadata_connection_type == 'sqlite':
            if dataset_location is None:
                raise ValueError("dataset location must be specified for sqlite mode")
            db_file = Path(dataset_location) / 'meta' / 'index.db'
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.sql_handler = SQLiteHandler(db_file)
            self.mode = "local"
            self.file_api = AtriumFileHandler(tsc_file_location)
            self.settings_dict = self._get_all_settings()

        elif metadata_connection_type == 'mysql' or metadata_connection_type == 'mariadb':
            host = connection_params['host']
            user = connection_params['user']
            password = connection_params['password']
            database = connection_params['database']
            port = connection_params['port']
            self.sql_handler = MariaDBHandler(host, user, password, database, port)
            self.mode = "local"
            self.file_api = AtriumFileHandler(tsc_file_location)
            self.settings_dict = self._get_all_settings()

        elif metadata_connection_type == 'api':
            self.mode = "api"
            self.api_url = api_url
            self.token = token

        else:
            raise ValueError("metadata_connection_type must be one of sqlite, mysql, mariadb or api")

        self._measures = self.get_all_measures()
        self._devices = self.get_all_devices()

        self._measure_ids = {}
        for measure_id, measure_info in self._measures.items():
            self._measure_ids[(measure_info['tag'], measure_info['freq_nhz'], measure_info['unit'])] = measure_id

        self._device_ids = {}
        for device_id, device_info in self._devices.items():
            self._device_ids[device_info['tag']] = device_id

    @classmethod
    def create_dataset(cls, dataset_location: Union[str, PurePath], database_type: str = None,
                       protected_mode: str = None, overwrite: str = None, connection_params: dict = None):
        """
        A class method to create a new dataset.

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

        :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the dataset will be written.
        :param str database_type: Specifies the type of metadata database to use. Options are "sqlite", "mysql", or "mariadb".
        :param str protected_mode: Specifies the protection mode of the metadata database. Allowed values are "True" or "False". If "True", data deletion will not be allowed. If "False", data deletion will be allowed. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.
        :param str overwrite: Specifies the behavior to take when new data being inserted overlaps in time with existing data. Allowed values are "error", "ignore", or "overwrite". Upon triggered overwrite: if "error", an error will be raised. If "ignore", the new data will not be inserted. If "overwrite", the old data will be overwritten with the new data. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.
        :param dict connection_params: A dictionary containing connection parameters for "mysql" or "mariadb" database type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.

        :return: An initialized AtriumSDK object.
        :rtype: AtriumSDK
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
            host = connection_params['host']
            user = connection_params['user']
            password = connection_params['password']
            database = connection_params['database']
            port = connection_params['port']
            MariaDBHandler(host, user, password, database, port).create_schema()

        sdk_object = cls(dataset_location=dataset_location, metadata_connection_type=database_type,
                         connection_params=connection_params)

        # Add settings
        sdk_object.sql_handler.insert_setting(PROTECTED_MODE_SETTING_NAME, str(protected_mode))
        sdk_object.sql_handler.insert_setting(OVERWRITE_SETTING_NAME, str(overwrite))

        sdk_object.settings_dict = sdk_object._get_all_settings()

        return sdk_object

    def _get_all_settings(self):
        settings = self.sql_handler.select_all_settings()
        return {setting[0]: setting[1] for setting in settings}

    def _overwrite_delete_data(self, measure_id, device_id, new_time_data, time_0, raw_time_type, values_size,
                               freq_nhz):
        auto_convert_gap_to_time_array = True
        return_intervals = False
        analog = False

        period_ns = int((10 ** 18) // freq_nhz)
        if raw_time_type == 1:
            # already a timestamp array.
            end_time_ns = int(new_time_data[-1])
        elif raw_time_type == 2:
            # Convert Gap array to timestamp array.
            end_time_ns = time_0 + (values_size * period_ns) + np.sum(new_time_data[1::2])
            new_time_data = np.arange(time_0, end_time_ns, period_ns, dtype=np.int64)
        else:
            raise ValueError("Overwrite only supported for gap arrays and timestamp arrays.")

        overwrite_file_dict = {}
        all_old_file_blocks = []
        old_block_list = self.get_block_id_list(measure_id, start_time_n=int(time_0),
                                                end_time_n=end_time_ns, device_id=device_id)
        _LOGGER.debug(f"old_block_list:{old_block_list} = self.get_block_id_list(measure_id={measure_id}, start_time_n={int(time_0)}, end_time_n={end_time_ns}, device_id={device_id})")

        old_file_id_dict = self.get_filename_dict(list(set([row[3] for row in old_block_list])))

        for file_id, filename in old_file_id_dict.items():
            file_block_list = self.sql_handler.select_blocks_from_file(file_id)
            all_old_file_blocks.extend(file_block_list)
            read_list = condense_byte_read_list(file_block_list)

            encoded_bytes = self.file_api.read_file_list_3(measure_id, read_list, old_file_id_dict)

            num_bytes_list = [row[5] for row in file_block_list]

            # start_time_n and end_time_n are only used to truncate the output of decode block arr.
            start_time_n = file_block_list[0][6]
            end_time_n = file_block_list[-1][7] * 2  # Just don't truncate output
            old_headers, old_times, old_values = \
                self.decode_block_arr(encoded_bytes, num_bytes_list, start_time_n, end_time_n, analog,
                                      auto_convert_gap_to_time_array, return_intervals)

            _LOGGER.debug(f"old_values: {old_values}")

            old_times = old_times.astype(np.int64)
            diff_mask = np.in1d(old_times, new_time_data, assume_unique=False, invert=True)

            if np.any(diff_mask):
                diff_times, diff_values = old_times[diff_mask], old_values[diff_mask]
                # Since all the headers are from the same file they should have the same scale factors
                # And data types.
                freq_nhz = old_headers[0].freq_nhz
                scale_m = old_headers[0].scale_m
                scale_b = old_headers[0].scale_b

                raw_value_type = old_headers[0].v_raw_type
                encoded_value_type = old_headers[0].v_encoded_type

                raw_time_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
                encoded_time_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

                encoded_bytes, encode_headers, byte_start_array = self.block.encode_blocks(
                    diff_times, diff_values, freq_nhz, diff_times[0],
                    raw_time_type=raw_time_type,
                    raw_value_type=raw_value_type,
                    encoded_time_type=encoded_time_type,
                    encoded_value_type=encoded_value_type,
                    scale_m=scale_m,
                    scale_b=scale_b)

                diff_filename = self.file_api.write_bytes(measure_id, device_id, encoded_bytes)

                block_data, interval_data = get_block_and_interval_data(
                    measure_id, device_id, encode_headers, byte_start_array, [])

                overwrite_file_dict[diff_filename] = (block_data, interval_data)

        return overwrite_file_dict, [row[0] for row in old_block_list], list(old_file_id_dict.items())

    def write_data(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq_nhz: int,
                   time_0: int, raw_time_type: int = None, raw_value_type: int = None, encoded_time_type: int = None,
                   encoded_value_type: int = None, scale_m: float = None, scale_b: float = None):
        """
        The advanced method for writing new data to the dataset. This method can be used, like in the example,
        to express time data as a gap array (even sized array, odd values are indices of value_data after a gap
        and even values are the durations of the corresponding gaps in nanoseconds)

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


        :param int measure_id: The measure identifier corresponding to the measures table in the linked
            relational database.
        :param int device_id: The device identifier corresponding to the devices table in the linked
            relational database.
        :param numpy.ndarray time_data: A 1D numpy array representing the time information of the data to be written.
        :param numpy.ndarray value_data: A 1D numpy array representing the value information of the data to be written.
        :param int freq_nhz: The sample frequency, in nanohertz, of the data to be written.
        :param int time_0: The start time of the data to be written.
        :param int raw_time_type: An identifer representing the time format being written, corresponding to the options
            written in the block header.
        :param int raw_value_type: An identifer representing the value format being written, corresponding to the
            options written in the block header.
        :param int encoded_time_type: An identifer representing how the time information is encoded, corresponding
            to the options written in the block header.
        :param int encoded_value_type: An identifer representing how the value information is encoded, corresponding
            to the options written in the block header.
        :param float scale_m: A constant factor to scale digital data to transform it to analog (None if raw data
            is already analog). The slope (m) in y = mx + b
        :param float scale_b: A constant factor to offset digital data to transform it to analog (None if raw data
            is already analog). The y-intercept (b) in y = mx + b

        :rtype: Tuple[numpy.ndarray, List[BlockMetadata], numpy.ndarray, str]
        :returns: A numpy byte array of the compressed blocks.\n
            A list of BlockMetadata objects representing the binary
            block headers.\n
            A 1D numpy array representing the byte locations of the start of
            each block.\n
            The filename of the written blocks.
        """
        assert self.mode == "local"
        assert value_data.size > 0, "Cannot write no data."
        # Apply Scale Factors and Convert
        # if scale_b is not None:
        #     value_data -= scale_b
        #
        # if scale_m is not None:
        #     value_data /= scale_m
        #     value_data = value_data.astype(np.int64)

        # Calculate New Intervals
        write_intervals = find_intervals(freq_nhz, raw_time_type, time_data, time_0, int(value_data.size))
        write_intervals_o = Intervals(write_intervals)

        # Get Current Intervals
        current_intervals = self.get_interval_array(
            measure_id, device_id=device_id, gap_tolerance_nano=0,
            start=int(write_intervals[0][0]), end=int(write_intervals[-1][-1]))

        current_intervals_o = Intervals(current_intervals)

        overwrite_file_dict, old_block_ids, old_file_list = None, None, None
        if current_intervals_o.intersection(write_intervals_o).duration() > 0:
            _LOGGER.debug(f"Overlap measure_id {measure_id}, device_id {device_id}, " 
                          f"existing intervals {current_intervals}, new intervals {write_intervals}")
            if OVERWRITE_SETTING_NAME not in self.settings_dict:
                raise ValueError("Overwrite detected, but overwrite behavior not set.")

            overwrite_setting = self.settings_dict[OVERWRITE_SETTING_NAME]
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

        block_data, interval_data = get_block_and_interval_data(
            measure_id, device_id, encode_headers, byte_start_array, write_intervals)

        if overwrite_file_dict is not None:
            # Add new data to sql insertion data.
            overwrite_file_dict[filename] = (block_data, interval_data)
            # Update SQL
            old_file_ids = [file_id for file_id, filename in old_file_list]
            _LOGGER.debug(f"{measure_id}, {device_id}): overwrite_file_dict: {overwrite_file_dict}\n old_block_ids: {old_block_ids}\n old_file_ids: {old_file_ids}\n")
            self.sql_handler.update_tsc_file_data(overwrite_file_dict, old_block_ids, old_file_ids)

            # Delete files
            # for file_id, filename in old_file_list:
            #     file_path = Path(self.file_api.to_abs_path(filename, measure_id, device_id))
            #     file_path.unlink(missing_ok=True)
        else:
            # Insert SQL Rows
            self.sql_handler.insert_tsc_file_data(filename, block_data, interval_data)

        return encoded_bytes, encode_headers, byte_start_array, filename

    def write_data_file_only(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray,
                             freq_nhz: int, time_0: int, raw_time_type: int = None, raw_value_type: int = None,
                             encoded_time_type: int = None, encoded_value_type: int = None, scale_m: float = None,
                             scale_b: float = None):
        """
        Advanced Function, Use with caution. Writes a tsc file to disk and returns relevant metadata needed to write to
        the metadata table at a later time. Useful for performing encoding and disk io from a worker process and then
        passing metadata to be written to sql table from main process.

        >>> import numpy as np
        >>> sdk = AtriumSDK(dataset_location='./example_dir')
        >>> # Create some time data.
        >>> time_data = np.arange(1234567890, 1234567890 + 3600, dtype=np.int64) * (10 ** 9)
        >>> # Create some value data of equal dimension.
        >>> value_data = np.sin(time_data)
        >>> # Encode Data and Write To Disk.
        >>> measure_id, device_id, filename, encode_headers, byte_start_array, intervals = sdk.write_data_file_only(measure_id=42, device_id=99, time_data=time_data, value_data=value_data, freq_nhz=1_000_000_000, time_0=int(time_data[0]), raw_time_type=1, raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_b=None, scale_m=None)
        >>> # Add to metadata sql table.
        >>> sdk.metadata_insert_sql(measure_id, device_id, filename, encode_headers, byte_start_array, intervals)

        :param measure_id: The measure identifier corresponding to the measures table in the linked relational database.
        :type measure_id: int
        :param device_id: The device identifier corresponding to the devices table in the linked relational database.
        :type device_id: int
        :param time_data: A 1D numpy array representing the time information of the data to be written.
        :type time_data: np.ndarray
        :param value_data: A 1D numpy array representing the value information of the data to be written.
        :type value_data: np.ndarray
        :param freq_nhz: The sample frequency, in nanohertz, of the data to be written.
        :type freq_nhz: int
        :param time_0: Start time of the data.
        :type time_0: int
        :param raw_time_type: Type of raw time data, default is None.
        :type raw_time_type: int, optional
        :param raw_value_type: Type of raw value data, default is None.
        :type raw_value_type: int, optional
        :param encoded_time_type: Type of encoded time data, default is None.
        :type encoded_time_type: int, optional
        :param encoded_value_type: Type of encoded value data, default is None.
        :type encoded_value_type: int, optional
        :param scale_m: A constant factor to scale digital data to transform it to analog
            (None if raw data is already analog). The slope (m) in y = mx + b, default is None.
        :type scale_m: float, optional
        :param scale_b: A constant factor to offset digital data to transform it to analog (None if raw data is already
            analog). The y-intercept (b) in y = mx + b, default is None.
        :type scale_b: float, optional
        :rtype: Tuple[int, int, str, list, np.ndarry, list]
        :returns: Measure identifier.\n
            Device identifier.\n
            TSC filename.\n
            TSC header list.\n
            Byte location of the start of each block.\n
            List of calculated contiguous data intervals.

        """
        assert self.mode == "local"

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

    def write_data_easy(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray,
                        freq: int, scale_m: float = None, scale_b: float = None, time_units: str = None,
                        freq_units: str = None):
        """
        The simplified method for writing new data to the dataset

        >>> import numpy as np
        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> new_measure_id = 21
        >>> new_device_id = 21
        >>> # Create some time data.
        >>> freq_hz = 1
        >>> time_data = np.arange(1234567890, 1234567890 + 3600, dtype=np.int64)
        >>> # Create some value data of equal dimension.
        >>> value_data = np.sin(time_data)
        >>> sdk.write_data_easy(measure_id=new_measure_id, device_id=new_device_id, time_data=time_data, value_data=value_data, freq=freq_hz, freq_units="Hz", time_units="s")

        :param int measure_id: The measure identifier corresponding to the measures table in the linked
            relational database.
        :param int device_id: The device identifier corresponding to the devices table in the linked
            relational database.
        :param numpy.ndarray time_data: A 1D numpy array representing the time information of the data to be written.
        :param numpy.ndarray value_data: A 1D numpy array representing the value information of the data to be written.
        :param int freq: The sample frequency, in nanohertz, of the data to be written. If you want to use units
            other than nanohertz specify the desired unit using the "freq_units" parameter.
        :param float scale_m: A constant factor to scale digital data to transform it to analog (None if raw data
            is already analog). The slope (m) in y = mx + b
        :param float scale_b: A constant factor to offset digital data to transform it to analog (None if raw data
            is already analog). The y-intercept (b) in y = mx + b
        :param str time_units: The unit used for the time data which can be one of ["s", "ms", "us", "ns"]. If units
            other than nanoseconds are used the time values will be converted to nanoseconds and then rounded to the
            nearest integer.
        :param str freq_units: The unit used for the specified frequency. This value can be one of ["nHz", "uHz", "mHz",
            "Hz", "kHz", "MHz"]. Keep in mind if you use extreemly large values for this it will be converted to nanohertz
            in the backend, and you may overflow 64bit integers.

        """
        time_units = "ns" if time_units is None else time_units
        freq_units = "nHz" if freq_units is None else freq_units

        # Check if they are using time units other than nanoseconds and if they are convert time_data to nanoseconds
        if time_units != "ns":
            time_data = convert_to_nanoseconds(time_data, time_units)

        if freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)

        if time_data.size == value_data.size:
            raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
        else:
            raw_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # gap_arr = create_gap_arr_fast(time_data, 1, freq_nhz)

        encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO
        if np.issubdtype(value_data.dtype, np.integer):

            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        self.write_data(measure_id, device_id, time_data, value_data, freq, int(time_data[0]), raw_time_type=raw_t_t,
                        raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                        scale_m=scale_m, scale_b=scale_b)

    def write_encounter(self, patient_id, device_id, start_time_n, end_time_n):
        pass

    def get_data_api(self, measure_id: int, start_time_n: int, end_time_n: int,
                     device_id: int = None, patient_id: int = None, mrn: int = None,
                     auto_convert_gap_to_time_array=True, return_intervals=False, analog=True):
        headers = {"Authorization": "Bearer {}".format(self.token)}

        block_info_url = self.get_block_info_api_url(measure_id, start_time_n, end_time_n, device_id, patient_id, mrn)

        block_info_response = requests.get(block_info_url, headers=headers)

        if not block_info_response.ok:
            raise block_info_response.raise_for_status()

        block_info_list = block_info_response.json()

        if len(block_info_list) == 0:
            return [], np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        block_requests = self.threaded_block_requests(block_info_list)

        for response in block_requests:
            if not response.ok:
                raise response.raise_for_status()

        encoded_bytes = np.concatenate(
            [np.frombuffer(response.content, dtype=np.uint8) for response in block_requests], axis=None)

        num_bytes_list = [row['num_bytes'] for row in block_info_list]

        headers, r_times, r_values = \
            self.decode_block_arr(encoded_bytes, num_bytes_list, start_time_n, end_time_n, analog,
                                  auto_convert_gap_to_time_array, return_intervals)

        return headers, r_times, r_values

    def threaded_block_requests(self, block_info_list):
        with ThreadPoolExecutor(max_workers=10) as executor:
            block_byte_list_threaded = list(
                executor.map(self.get_block_bytes_response, [row['id'] for row in block_info_list]))
        return block_byte_list_threaded

    def block_session_requests(self, block_info_list):
        session = Session()
        session.headers = {"Authorization": "Bearer {}".format(self.token)}
        block_byte_list_2 = \
            [self.get_block_bytes_response_from_session(row['id'], session) for row in block_info_list]
        return block_byte_list_2

    def threadless_block_requests(self, block_info_list):
        block_byte_list = [self.get_block_bytes_response(row['id']) for row in block_info_list]
        return block_byte_list

    def get_block_bytes_response(self, block_id):
        headers = {"Authorization": "Bearer {}".format(self.token)}
        block_request_url = self.api_url + "/v1/sdk/blocks/{}".format(block_id)
        return requests.get(block_request_url, headers=headers)

    def get_block_bytes_response_from_session(self, block_id, session: Session):
        block_request_url = self.api_url + "/v1/sdk/blocks/{}".format(block_id)
        return session.get(block_request_url)

    def get_block_info_api_url(self, measure_id, start_time_n, end_time_n, device_id, patient_id, mrn):
        if device_id is not None:
            block_info_url = \
                self.api_url + "/v1/sdk/blocks/?start_time={}&end_time={}&measure_id={}&device_id={}".format(
                    start_time_n, end_time_n, measure_id, device_id)
        elif patient_id is not None:
            block_info_url = \
                self.api_url + "/v1/sdk/blocks/?start_time={}&end_time={}&measure_id={}&patient_id={}".format(
                    start_time_n, end_time_n, measure_id, patient_id)

        elif mrn is not None:
            block_info_url = \
                self.api_url + "/v1/sdk/blocks/?start_time={}&end_time={}&measure_id={}&mrn={}".format(
                    start_time_n, end_time_n, measure_id, mrn)
        else:
            raise ValueError("One of [device_id, patient_id, mrn] must be specified.")
        return block_info_url

    def get_batched_data_generator(self, measure_id: int, start_time_n: int = None, end_time_n: int = None,
                                   device_id: int = None, patient_id=None, auto_convert_gap_to_time_array=True,
                                   return_intervals=False, analog=True, block_info=None, max_kbyte_in_memory=None,
                                   window_size=None, step_size=None, get_last_window=True):
        """
        Generates batched data from the dataset.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> measure_id = 123
        >>> start_time_n = 1000000000
        >>> end_time_n = 2000000000
        >>> device_id = 456
        >>> patient_id = 789
        >>> max_kbyte_in_memory = 1000
        >>> for total_query_index, times, values in sdk.get_batched_data_generator(measure_id=measure_id, start_time_n=start_time_n, end_time_n=end_time_n, device_id=device_id, patient_id=patient_id, max_kbyte_in_memory=max_kbyte_in_memory):
            ...     print(f"total_query_index: {total_query_index}, times: {times}, values: {values}")

        :param int measure_id: The measure identifier corresponding to the measures table in the linked relational database.
        :param int start_time_n: The start time of the data to be retrieved, in nanohertz.
        :param int end_time_n: The end time of the data to be retrieved, in nanohertz.
        :param int device_id: The device identifier corresponding to the devices table in the linked relational database.
        :param patient_id: The patient identifier.
        :param bool auto_convert_gap_to_time_array: If True, convert gap in time data to a numpy array.
        :param bool return_intervals: If True, return intervals instead of values.
        :param bool analog: If True, return analog data.
        :param dict block_info: A dictionary containing information about blocks.
        :param int max_kbyte_in_memory: The maximum amount of memory to use, in kilobytes.
        :param int window_size: The size of each batch, in number of values.
        :param int step_size: The step size between each batch, in number of values.
        :param bool get_last_window: If True, return the last window, even if its size is less than `window_size`.

        :returns: generator object
            Each element is a tuple (int, numpy.ndarray, numpy.ndarray) representing:
            The starting index of the current batch.
            A 1D numpy array of time information.
            A 1D numpy array of value information.
        """
        if window_size is not None:
            if step_size is None:
                step_size = window_size

        if block_info is None:

            block_list = self.get_block_id_list(int(measure_id), start_time_n=start_time_n, end_time_n=end_time_n,
                                                device_id=device_id, patient_id=patient_id)

            file_id_list = list(set([row['file_id'] for row in block_list]))
            filename_dict = self.get_filename_dict(file_id_list)

        else:
            block_list = block_info['block_list']
            filename_dict = block_info['filename_dict']

        if len(block_list) == 0:
            return

        current_memory_kb = 0
        cur_values = 0
        current_index = 0
        current_blocks_meta = []
        remaining_values = sum([block_metadata['num_values'] for block_metadata in block_list])
        times_before, values_before = None, None
        for block_metadata in block_list:
            current_blocks_meta.append(block_metadata)
            current_memory_kb += (block_metadata['num_bytes'] +
                                  (block_metadata['num_values'] * 16)) / 1000
            cur_values += block_metadata['num_values']
            remaining_values -= block_metadata['num_values']

            if current_memory_kb >= max_kbyte_in_memory and (window_size is None or cur_values >= window_size):
                headers, r_times, r_values = self.get_blocks(
                    current_blocks_meta, filename_dict, measure_id, start_time_n, end_time_n, analog,
                    auto_convert_gap_to_time_array, return_intervals,
                    times_before=times_before, values_before=values_before)

                yield from yield_data(r_times, r_values, window_size, step_size,
                                      get_last_window and (current_blocks_meta[-1] is block_list[-1]), current_index)

                current_index += r_values.size

                if window_size is not None:
                    next_step = (((r_values.size - window_size) // step_size) + 1) * step_size
                    times_before, values_before = r_times[next_step:], r_values[next_step:]
                    current_index -= values_before.size

                del headers, r_times, r_values

                current_memory_kb = 0
                cur_values = 0
                current_blocks_meta = []

        if current_memory_kb > 0:
            headers, r_times, r_values = self.get_blocks(
                current_blocks_meta, filename_dict, measure_id, start_time_n, end_time_n, analog,
                auto_convert_gap_to_time_array, return_intervals,
                times_before=times_before, values_before=values_before)
            if window_size is not None and r_values.size < window_size:
                if get_last_window:
                    current_index += r_values.size
                    last_num_values = 0
                    last_blocks_meta = [block_list[-1]]
                    for block_metadata in reversed(block_list[:-1]):
                        last_blocks_meta = [block_metadata] + last_blocks_meta
                        last_num_values += block_metadata['num_values']
                        if last_num_values >= window_size:
                            break

                    headers, r_times, r_values = self.get_blocks(
                        last_blocks_meta, filename_dict, measure_id, start_time_n, end_time_n, analog,
                        auto_convert_gap_to_time_array, return_intervals)

                    r_times, r_values = r_times[-window_size:], r_values[-window_size:]
                    current_index -= window_size

                    if r_values.size == window_size:
                        yield from yield_data(r_times, r_values, window_size, step_size, False, current_index)
            else:
                yield from yield_data(r_times, r_values, window_size, step_size, get_last_window, current_index)

    def get_blocks(self, current_blocks_meta, filename_dict, measure_id, start_time_n, end_time_n, analog,
                   auto_convert_gap_to_time_array, return_intervals, times_before=None, values_before=None):
        read_list = condense_byte_read_list(current_blocks_meta)
        encoded_bytes = self.file_api.read_file_list_3(measure_id, read_list, filename_dict)
        num_bytes_list = [row[5] for row in current_blocks_meta]
        headers, r_times, r_values = self.decode_block_arr(
            encoded_bytes, num_bytes_list, start_time_n, end_time_n, analog, auto_convert_gap_to_time_array,
            return_intervals, times_before=times_before, values_before=values_before)
        return headers, r_times, r_values

    def get_block_info(self, measure_id: int, start_time_n: int = None, end_time_n: int = None, device_id: int = None,
                       patient_id=None):
        block_list = self.get_block_id_list(int(measure_id), start_time_n=int(start_time_n), end_time_n=int(end_time_n),
                                            device_id=device_id, patient_id=patient_id)

        read_list = condense_byte_read_list(block_list)

        file_id_list = [row[1] for row in read_list]

        filename_dict = self.get_filename_dict(file_id_list)

        return {'block_list': block_list, 'filename_dict': filename_dict}

    def get_data(self, measure_id: int, start_time_n: int = None, end_time_n: int = None, device_id: int = None,
                 patient_id=None, auto_convert_gap_to_time_array=True, return_intervals=False, analog=True,
                 block_info=None, time_units: str = "ns"):
        """
        The method for querying data from the dataset, indexed by signal type (measure_id),
        time (start_time_n and end_time_n) and data source (device_id and patient_id)

        >>> start_epoch_s = 1669668855
        >>> end_epoch_s = start_epoch_s + 3600  # 1 hour after start.
        >>> start_epoch_nano = start_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
        >>> end_epoch_nano = end_epoch_s * (10 ** 9)  # Convert seconds to nanoseconds
        >>> _, r_times, r_values = sdk.get_data(measure_id=1, start_time_n=start_epoch_s, end_time_n=end_epoch_nano, device_id=4)
        >>> r_times
        array([1669668855000000000, 1669668856000000000, 1669668857000000000, ...,
        1669672452000000000, 1669672453000000000, 1669672454000000000],
        dtype=int64)
        >>> r_values
        array([ 0.32731968,  0.79003189,  0.99659552, ..., -0.59080797,
        -0.93542358, -0.97675089])

        :param block_info:
        :param int measure_id: The measure identifier corresponding to the measures table in the
            linked relational database.
        :param int start_time_n: The start epoch in nanoseconds of the data you would like to query.
        :param int end_time_n: The end epoch in nanoseconds of the data you would like to query.
        :param int device_id: The device identifier corresponding to the devices table in the
            linked relational database.
        :param int patient_id: The patient identifier corresponding to the encounter table in the
            linked relational database.
        :param bool auto_convert_gap_to_time_array: If the raw time format type is type gap array,
            automatically convert returned time data as an array of timestamps.
        :param bool return_intervals: Automatically convert time return type to time intervals.
        :param bool analog: Automatically convert value return type to analog signal.
        :param sqlalchemy.engine.Connection connection: You can pass in an sqlalchemy connection object from the
            relational database if you already have one open.
        :param str time_units: If you would like the time array returned in units other than nanoseconds you can
            choose from one of ["s", "ms", "us", "ns"].

        :rtype: Tuple[List[BlockMetadata], numpy.ndarray, numpy.ndarray]
        :returns: A list of the block header python objects.\n
            A numpy 1D array representing the time data (usually an array of timestamps).\n
            A numpy 1D array representing the value data.

        """
        # check that a correct unit type was entered
        time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}

        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        # convert start and end time to nanoseconds
        start_time_n = int(start_time_n * time_unit_options[time_units])
        end_time_n = int(end_time_n * time_unit_options[time_units])

        _LOGGER.debug("\n")
        start_bench_total = time.perf_counter()

        if self.mode == "api":
            self.get_data_api(measure_id, start_time_n, end_time_n,
                              device_id=device_id, patient_id=patient_id,
                              auto_convert_gap_to_time_array=auto_convert_gap_to_time_array,
                              return_intervals=return_intervals, analog=analog)

        elif self.mode == "local":
            if block_info is None:
                start_bench = time.perf_counter()
                block_list = self.get_block_id_list(int(measure_id), start_time_n=int(start_time_n),
                                                    end_time_n=int(end_time_n), device_id=device_id,
                                                    patient_id=patient_id)
                end_bench = time.perf_counter()
                _LOGGER.debug(f"get block info {(end_bench - start_bench) * 1000} ms")

                # read_list = condense_byte_read_list([row[2:6] for row in block_list])
                start_bench = time.perf_counter()
                read_list = condense_byte_read_list(block_list)
                end_bench = time.perf_counter()
                _LOGGER.debug(f"condense read list {(end_bench - start_bench) * 1000} ms")

                # if no matching block ids
                if len(read_list) == 0:
                    return [], np.array([]), np.array([])

                start_bench = time.perf_counter()
                file_id_list = [row[1] for row in read_list]

                filename_dict = self.get_filename_dict(file_id_list)
                end_bench = time.perf_counter()
                _LOGGER.debug(f"get filename dictionary  {(end_bench - start_bench) * 1000} ms")

            else:
                block_list = block_info['block_list']
                filename_dict = block_info['filename_dict']

                # if no matching block ids
                if len(block_list) == 0:
                    return [], np.array([]), np.array([])

            headers, r_times, r_values = self.get_data_from_blocks(block_list, filename_dict, measure_id, start_time_n,
                                                                   end_time_n, return_intervals, analog,
                                                                   auto_convert_gap_to_time_array)

            end_bench_total = time.perf_counter()
            # print(f"Total get data call took {round(end_bench_total - start_bench_total, 2)}: {r_values.size} values")
            # print(f"{round(r_values.size / (end_bench_total-start_bench_total), 2)} values per second.")
            # start_bench = time.perf_counter()
            # left, right = bisect.bisect_left(r_times, start_time_n), bisect.bisect_left(r_times, end_time_n)
            # r_times, r_values = r_times[left:right], r_values[left:right]
            # end_bench = time.perf_counter()
            # _LOGGER.debug(f"truncate data {(end_bench - start_bench) * 1000} ms")

            # convert time data from nanoseconds to unit of choice
            r_times = r_times / time_unit_options[time_units]

            if np.all(r_times == np.floor(r_times)):
                r_times = r_times.astype('int64')

            return headers, r_times, r_values

    def get_data_from_blocks(self, block_list, filename_dict, measure_id, start_time_n, end_time_n,
                             return_intervals=False, analog=True, auto_convert_gap_to_time_array=True):
        start_bench = time.perf_counter()
        read_list = condense_byte_read_list(block_list)
        # File Read Method 1
        # encoded_bytes = self.file_api.read_file_list_1(measure_id, read_list, filename_dict)
        # File Read Method 2 Not Working
        # encoded_bytes = self.file_api.read_file_list_2(measure_id, read_list, filename_dict)
        # File Read Method 3
        encoded_bytes = self.file_api.read_file_list_3(measure_id, read_list, filename_dict)
        end_bench = time.perf_counter()
        _LOGGER.debug(f"read from disk {(end_bench - start_bench) * 1000} ms")
        num_bytes_list = [row[5] for row in block_list]
        headers, r_times, r_values = \
            self.decode_block_arr(encoded_bytes, num_bytes_list, start_time_n, end_time_n, analog,
                                  auto_convert_gap_to_time_array, return_intervals)
        return headers, r_times, r_values

    def decode_block_arr(self, encoded_bytes, num_bytes_list, start_time_n, end_time_n, analog,
                         auto_convert_gap_to_time_array, return_intervals, times_before=None, values_before=None):
        start_bench = time.perf_counter()
        byte_start_array = np.cumsum(num_bytes_list, dtype=np.uint64)
        byte_start_array = np.concatenate([np.array([0], dtype=np.uint64), byte_start_array[:-1]], axis=None)
        end_bench = time.perf_counter()
        _LOGGER.debug(f"arrange block info {(end_bench - start_bench) * 1000} ms")

        r_times, r_values, headers = self.block.decode_blocks(
            encoded_bytes, byte_start_array, analog=analog, times_before=times_before, values_before=values_before)

        new_times_index = 0 if times_before is None else times_before.size
        new_values_index = 0 if values_before is None else values_before.size

        if auto_convert_gap_to_time_array and \
                all([h.t_raw_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO for h in headers]):

            if return_intervals:
                # This whole block almost never runs, but it probably bug riddled.

                intervals = []
                gap_arr = r_times[new_times_index:].reshape((-1, 2))

                cur_gap = 0

                # create a list of intervals from all blocks
                for h in headers:
                    intervals.extend(
                        convert_gap_array_to_intervals(h.start_n,
                                                       gap_arr[cur_gap:cur_gap + h.num_gaps],
                                                       h.num_vals,
                                                       h.freq_nhz))

                    cur_gap += h.num_gaps

                intervals = np.array(intervals, dtype=np.int64)

                # Sort by start time
                interval_order = np.argsort(intervals.T[0])

                sorted_intervals = intervals[interval_order]

                # Sort the value array

                # Get arr of original value order [[start_ind, num_values], ...]
                org_value_order_arr = np.zeros((intervals.shape[0], 2), dtype=np.uint64)
                cur_index = 0

                for block_i, block_num_values in enumerate(intervals.T[2]):
                    org_value_order_arr[block_i] = (cur_index, block_num_values)
                    cur_index += block_num_values

                # Populate new value array in correct order.
                sorted_values = np.zeros(r_values[new_values_index:].shape, dtype=r_values.dtype)
                cur_index = 0
                for org_index, block_num_values in org_value_order_arr[interval_order]:
                    org_index = int(org_index)
                    block_num_values = int(block_num_values)
                    sorted_values[cur_index:cur_index + block_num_values] = \
                        r_values[new_values_index:][org_index:org_index + block_num_values]

                    cur_index += block_num_values

                r_times, r_values = sorted_intervals, sorted_values

            else:
                # r_times, r_values[new_values_index:] = self.filter_gap_data_to_timestamps(
                #     end_time_n, headers, r_times[new_times_index:], r_values[new_values_index:],
                #     start_time_n, times_before=times_before)

                r_times, new_values = self.filter_gap_data_to_timestamps(
                    end_time_n, headers, r_times[new_times_index:], r_values[new_values_index:],
                    start_time_n, times_before=times_before)

                r_values[new_values_index:new_values_index + new_values.size] = new_values
                r_values = r_values[:new_values_index + new_values.size]

        else:
            r_times[new_times_index:], r_values[new_values_index:] = sort_data(
                r_times[new_times_index:], r_values[new_values_index:], headers)

        if times_before is not None:
            r_times[:times_before.size] = times_before

        if values_before is not None:
            r_values[:values_before.size] = values_before

        # Truncate data to match query.
        start_bench = time.perf_counter()
        left, right = bisect.bisect_left(r_times, start_time_n), bisect.bisect_left(r_times, end_time_n)
        r_times, r_values = r_times[left:right], r_values[left:right]
        end_bench = time.perf_counter()
        _LOGGER.debug(f"truncate data {(end_bench - start_bench) * 1000} ms")

        return headers, r_times, r_values

    def get_filename_dict(self, file_id_list):
        result_dict = {}
        for row in self.sql_handler.select_files(file_id_list):
            result_dict[row[0]] = row[1]

        return result_dict

    @staticmethod
    def filter_gap_data_to_timestamps(end_time_n, headers, r_times, r_values, start_time_n, times_before=None):
        start_bench = time.perf_counter()
        new_times_index = 0 if times_before is None else times_before.size
        is_int_times = all([(10 ** 18) % h.freq_nhz == 0 for h in headers])
        time_dtype = np.int64 if is_int_times else np.float64
        full_timestamps = np.zeros(r_values.size + new_times_index, dtype=time_dtype)
        cur_index, cur_gap = 0, 0
        if is_int_times:
            for block_i, h in enumerate(headers):
                period_ns = freq_nhz_to_period_ns(h.freq_nhz)
                full_timestamps[new_times_index:][cur_index:cur_index + h.num_vals] = \
                    np.arange(h.start_n, h.start_n + (h.num_vals * period_ns), period_ns)

                for _ in range(h.num_gaps):
                    full_timestamps[new_times_index:][cur_index + r_times[cur_gap]:cur_index + h.num_vals] += \
                        r_times[cur_gap + 1]
                    cur_gap += 2

                cur_index += h.num_vals
        else:
            for block_i, h in enumerate(headers):
                period_ns = float(10 ** 18) / float(h.freq_nhz)
                full_timestamps[new_times_index:][cur_index:cur_index + h.num_vals] = \
                    np.linspace(h.start_n, h.start_n + (h.num_vals * period_ns), num=h.num_vals, endpoint=False)

                for _ in range(h.num_gaps):
                    full_timestamps[new_times_index:][cur_index + r_times[cur_gap]:cur_index + h.num_vals] += \
                        r_times[cur_gap + 1]
                    cur_gap += 2

                cur_index += h.num_vals

        end_bench = time.perf_counter()
        _LOGGER.debug(f"Expand Gap Data {(end_bench - start_bench) * 1000} ms")
        # full_timestamps[new_times_index:], r_values = \
        #     sort_data(full_timestamps[new_times_index:], r_values, headers)

        sorted_times, sorted_values = sort_data(full_timestamps[new_times_index:], r_values, headers)
        full_timestamps[new_times_index:new_times_index + sorted_times.size], r_values = sorted_times, sorted_values

        return full_timestamps[:new_times_index + sorted_times.size], r_values

    def metadata_insert_sql(self, measure_id: int, device_id: int, path: str, metadata: list, start_bytes: np.ndarray,
                            intervals: list):

        block_data, interval_data = get_block_and_interval_data(
            measure_id, device_id, metadata, start_bytes, intervals)

        self.sql_handler.insert_tsc_file_data(path, block_data, interval_data)

    def get_interval_array(self, measure_id, device_id=None, patient_id=None, gap_tolerance_nano: int = None,
                           start=None, end=None):
        """
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
        :rtype: numpy.ndarray
        :returns: A 2D array representing the availability of a specified measure.

        """
        gap_tolerance_nano = 0 if gap_tolerance_nano is None else gap_tolerance_nano

        interval_result = self.sql_handler.select_intervals(
            measure_id, start_time_n=start, end_time_n=end, device_id=device_id, patient_id=patient_id)

        arr = []
        for row in interval_result:
            if len(arr) > 0 and row[3] - arr[-1][-1] <= gap_tolerance_nano:
                arr[-1][-1] = row[4]
            else:
                arr.append([row[3], row[4]])

        return np.array(arr, dtype=np.int64)

    def get_combined_intervals(self, measure_id_list, device_id=None, patient_id=None, gap_tolerance_nano: int = None,
                               start=None, end=None):
        if len(measure_id_list) == 0:
            return np.array([[]])
        result = self.get_interval_array(measure_id_list[0], device_id=device_id, patient_id=patient_id,
                                         gap_tolerance_nano=gap_tolerance_nano, start=start, end=end)

        for measure_id in measure_id_list[1:]:
            result = merge_interval_lists(
                result,
                self.get_interval_array(measure_id, device_id=device_id, patient_id=patient_id,
                                        gap_tolerance_nano=gap_tolerance_nano, start=start, end=end))

        return result

    def get_block_id_list(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        return self.sql_handler.select_blocks(measure_id, start_time_n, end_time_n, device_id, patient_id)

    def get_freq(self, measure_id: int, freq_units: str = None):
        """
        Returns the frequency of the signal corresponding to the specified measure_id.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> measure_id = 1
        >>> freq_units = "Hz"
        >>> frequency = sdk.get_freq(measure_id, freq_units)
        >>> print(frequency)
        10.0

        >>> freq = sdk.get_freq(measure_id=1, freq_units="nHz")
        >>> print(freq)
        10000000000

        :param int measure_id: The measure identifier corresponding to the measures table in the
            linked relational database.
        :param str freq_units: The units of the frequency to be returned.
        :rtype: float
        :return: The frequency in hertz.

        """
        if freq_units is None:
            freq_units = "nHz"

        measure_tuple = self.sql_handler.select_measure(measure_id=measure_id)

        if measure_tuple is None:
            raise ValueError(f"measure id {measure_id} not in sdk.")

        return convert_from_nanohz(measure_tuple[3], freq_units)

    def get_measure_tag(self, measure_id):
        pass

    def get_all_devices(self):
        """
        Retrieve information about all devices in the linked relational database.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> all_devices = sdk.get_all_devices()
        >>> print(all_devices)
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
        device_tuple_list = self.sql_handler.select_all_devices()
        device_dict = {}
        for device_id, device_tag, device_name, device_manufacturer, device_model, device_type, device_bed_id, \
                device_source_id in device_tuple_list:
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

        return device_dict

    def get_all_measures(self):
        """
        Retrieve information about all measures in the linked relational database.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> all_measures = sdk.get_all_measures()
        >>> print(all_measures)
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
        measure_tuple_list = self.sql_handler.select_all_measures()
        measure_dict = {}
        for measure_id, measure_tag, measure_name, measure_freq_nhz, measure_code, measure_unit, measure_unit_label, \
                measure_unit_code, measure_source_id in measure_tuple_list:
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

    def search_devices(self, tag_match=None, name_match=None):
        """
        Retrieve information about all devices in the linked relational database that match the specified search criteria.

        :param tag_match: A string to match against the `device_tag` field. If not None, only devices with a `device_tag`
            field containing this string will be returned.
        :type tag_match: str
        :param name_match: A string to match against the `device_name` field. If not None, only devices with a `device_name`
            field containing this string will be returned.
        :type name_match: str
        :return: A dictionary containing information about each device that matches the specified search criteria, including
            its id, tag, name, manufacturer, model, type, bed_id, and source_id.
        :rtype: dict
        """
        all_devices = self.get_all_devices()
        result = {}
        for device_id, device_info in all_devices.items():
            match_bool_list = [
                tag_match is None or tag_match in device_info['tag'],
                name_match is None or name_match in device_info['name']
            ]
            if all(match_bool_list):
                result[device_id] = device_info
        return result

    def search_measures(self, tag_match=None, freq_nhz=None, unit=None, name_match=None):
        """
        Retrieve information about all measures in the linked relational database that match the specified search criteria.

        :param tag_match: A string to match against the `measure_tag` field. If not None, only measures with a `measure_tag`
            field containing this string will be returned.
        :type tag_match: str
        :param freq_nhz: A value to match against the `measure_freq_nhz` field. If not None, only measures with a
            `measure_freq_nhz` field equal to this value will be returned.
        :type freq_nhz: int
        :param unit: A string to match against the `measure_unit` field. If not None, only measures with a `measure_unit`
            field equal to this string will be returned.
        :type unit: str
        :param name_match: A string to match against the `measure_name` field. If not None, only measures with a
            `measure_name` field containing this string will be returned.
        :type name_match: str
        :return: A dictionary containing information about each measure that matches the specified search criteria, including
            its id, tag, name, sample frequency (in nanohertz), code, unit, unit label, unit code, and source_id.
        :rtype: dict
        """
        all_measures = self.get_all_measures()
        result = {}
        for measure_id, measure_info in all_measures.items():
            match_bool_list = [
                tag_match is None or tag_match in measure_info['tag'],
                freq_nhz is None or freq_nhz == measure_info['freq_nhz'],
                unit is None or unit == measure_info['unit'],
                name_match is None or name_match in measure_info['name']
            ]
            if all(match_bool_list):
                result[measure_id] = measure_info
        return result

    def get_all_patient_ids(self, start=None, end=None):
        return [row[0] for row in self.sql_handler.select_all_patients_in_list()]

    def get_available_measures(self, device_id=None, patient_id=None, start=None, end=None):
        pass

    def get_available_devices(self, measure_id, start=None, end=None):
        pass

    def get_random_window(self, time_intervals, time_window_size_nano=30_000_000_000):
        # Get large enough interval
        start, end = random.choice(time_intervals)

        i = 0
        while end - start < time_window_size_nano:
            start, end = random.choice(time_intervals)
            i += 1
            if i > 9:
                return None, None

        # return a random interval, time_window_size_nano long in between start and end.

        # Generate random start
        start = random.randint(start, end - time_window_size_nano)

        # Calculate end
        end = start + time_window_size_nano

        return start, end

    def get_random_data(self, measure_id=None, device_list=None, start=None, end=None,
                        time_window_size_nano=30_000_000_000, gap_tolerance_nano=0):
        if device_list is None:
            device_list = self.get_available_devices(measure_id, start=start, end=end)

        if len(device_list) == 0:
            return np.array([]), np.array([])

        try_start, try_end = 0, 0
        i = 0
        device_id = None
        while try_end - try_start < time_window_size_nano:
            device_id = random.choice(device_list)
            interval_array = \
                self.get_interval_array(measure_id, device_id, gap_tolerance_nano=gap_tolerance_nano, start=start,
                                        end=end)

            interval_array = interval_array[(interval_array.T[1] - interval_array.T[0]) > time_window_size_nano]

            if len(interval_array) == 0:
                continue

            try_start, try_end = random.choice(interval_array)
            i += 1

            if i > 9:
                return np.array([]), np.array([])

        # Generate random start
        try_start = random.randint(try_start, try_end - time_window_size_nano)

        # Calculate end
        try_end = try_start + time_window_size_nano

        headers, r_times, r_values = self.get_data(measure_id, try_start, try_end, device_id)
        info = {'start': try_start, 'end': try_end, 'device_id': device_id}
        return info, r_times, r_values

    def create_derived_variable(self, function_list, args_list, kwargs_list,
                                dest_sdk=None, dest_measure_id_list=None, dest_device_id_list=None,
                                measure_id=None, device_id=None, start=None, end=None):
        for var in [measure_id, device_id, start, end]:
            if var is None:
                raise ValueError("[measure_id, device_id, start, end] must all be specified")

        dest_measure_id_list = [None for _ in range(len(function_list))] if \
            dest_measure_id_list is None else dest_measure_id_list
        dest_device_id_list = [None for _ in range(len(function_list))] if \
            dest_device_id_list is None else dest_device_id_list

        headers, intervals, values = self.get_data(measure_id, start, end, device_id,
                                                   auto_convert_gap_to_time_array=True, return_intervals=True,
                                                   analog=False)

        results = []

        for func, args, kwargs, dest_measure_id, dest_device_id in \
                zip(function_list, args_list, kwargs_list, dest_measure_id_list, dest_device_id_list):
            res = func(values, *args, **kwargs)
            results.append((intervals, res))
            if dest_sdk is not None:
                dest_measure_id = measure_id if dest_measure_id is None else dest_measure_id
                dest_device_id = device_id if dest_device_id is None else dest_device_id

                dest_sdk.auto_write_interval_data(dest_measure_id, dest_device_id, intervals, values, None, None, None)

        return results

    def auto_write_interval_data(self, measure_id, device_id, intervals, values, freq_nhz, scale_b, scale_m):
        gap_arr = convert_intervals_to_gap_array(intervals)

        t_t = 2
        if np.issubdtype(values.dtype, np.integer):
            raw_v_t = 1
            encoded_v_t = 3
        else:
            raw_v_t = 2
            encoded_v_t = 2

        self.write_data(measure_id, device_id, gap_arr.reshape((-1,)), values, freq_nhz, int(intervals[0][0]),
                        raw_time_type=t_t, raw_value_type=raw_v_t, encoded_time_type=t_t,
                        encoded_value_type=encoded_v_t, scale_m=scale_m, scale_b=scale_b)

    def insert_measure(self, measure_tag: str, freq: Union[int, float], units: str = None, freq_units: str = None,
                       measure_name: str = None):
        """
        Defines a new signal type to be stored in the dataset, as well as defining metadata related to the signal.

        freq and freq_units are required information, but it is also recommended to define a measure_tag
        (which can be done by specifying measure_tag as an optional parameter).

        The other optional parameters are measure_name (A description of the signal) and units
        (the units of the signal).

        >>> # Define a new signal.
        >>> freq = 500
        >>> freq_units = "Hz"
        >>> measure_tag = "ECG Lead II - 500 Hz"
        >>> measure_name = "Electrocardiogram Lead II Configuration 500 Hertz"
        >>> units = "mV"
        >>> measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units, measure_name=measure_name)

        :param freq: The sample frequency of the signal.
        :param str optional freq_units: The unit used for the specified frequency. This value can be one of ["Hz",
            "kHz", "MHz"]. Keep in mind if you use extremely large values for this it will be
            converted to Hertz in the backend, and you may overflow 64bit integers.
        :param str optional measure_tag: A unique string identifying the signal.
        :param str optional measure_name: A long form description of the signal.
        :param str optional units: The units of the signal.

        """

        # confirm the values are being entered as strings or none
        assert isinstance(measure_tag, str)
        assert isinstance(measure_name, str) or measure_name is None
        assert isinstance(units, str) or units is None

        freq_units = "nHz" if freq_units is None else freq_units

        if freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)

        if (measure_tag, freq, units) in self._measure_ids:
            return self._measure_ids[(measure_tag, freq, units)]

        return self.sql_handler.insert_measure(measure_tag, freq, units, measure_name)

    def insert_device(self, device_tag: str, device_name: str = None):
        """
        Defines a new source to be stored in the dataset, as well as defining metadata related to the source.

        device_id is required information, but it is also recommended to define a device_tag
        (which can be done by specifying device_tag as an optional parameter).

        The other optional parameter is device_name (A description of the source).

        >>> # Define a new source.
        >>> device_tag = "Monitor A3"
        >>> device_name = "Philips Monitor A3 in Room 2B"
        >>> new_device_id = sdk.insert_device(device_tag=device_tag, device_name=device_name)

        :param int device_id: A number identifying a unique source.
        :param str device_tag: A unique string identifying the source.
        :param str device_name: A long form description of the source.

        """

        if device_tag in self._device_ids:
            return self._device_ids[device_tag]

        return self.sql_handler.insert_device(device_tag, device_name)

    def measure_device_start_time_exists(self, measure_id, device_id, start_time_nano):
        """
        Check if a time interval for a measure, device and start_time already exists in the linked relational database.

        :param int measure_id: The identifier of the measure to check.
        :param int device_id: The identifier of the device to check.
        :param int start_time_nano: The start time of the interval, in nanoseconds.

        :return: True if the interval exists, False otherwise.
        :rtype: bool

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> measure_id = 3
        >>> device_id = 2
        >>> start_time_nano = 1234567890123
        >>> sdk.measure_device_start_time_exists(measure_id, device_id, start_time_nano)
        True
        """

        return self.sql_handler.interval_exists(measure_id, device_id, start_time_nano)

    def get_measure_id(self, measure_tag: str, freq: Union[int, float], units: str = None, freq_units: str = None):
        """
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
        7
        """
        units = "" if units is None else units
        freq_units = "nHz" if freq_units is None else freq_units
        freq_nhz = convert_to_nanohz(freq, freq_units)

        if (measure_tag, freq_nhz, units) in self._measure_ids:
            return self._measure_ids[(measure_tag, freq_nhz, units)]

        row = self.sql_handler.select_measure(measure_tag=measure_tag, freq_nhz=freq_nhz, units=units)
        if row is None:
            return None

        measure_id = row[0]
        self._measure_ids[(measure_tag, freq_nhz, units)] = measure_id
        return measure_id

    def get_measure_info(self, measure_id: int):
        """
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
        >>> print(measure_info)
        {
            'id': 1,
            'tag': 'Heart Rate',
            'name': 'Heart rate in beats per minute',
            'freq_nhz': 60,
            'code': 'HR',
            'unit': 'bpm',
            'unit_label': 'beats per minute',
            'unit_code': 'bpm',
            'source_id': 1
        }
        """

        if measure_id in self._measures:
            return self._measures[measure_id]

        row = self.sql_handler.select_measure(measure_id=measure_id)

        if row is None:
            return None

        measure_id, measure_tag, measure_name, measure_freq_nhz, measure_code, measure_unit, measure_unit_label, \
            measure_unit_code, measure_source_id = row

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

        self._measures[measure_id] = measure_info

        return measure_info

    def get_device_id(self, device_tag: str):
        """
        Retrieve the identifier of a device in the linked relational database based on its tag.

        :param str device_tag: The tag of the device to retrieve the identifier for.

        :return: The identifier of the device.
        :rtype: int

        >>> # Connect to example_dataset
        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>>
        >>> # Retrieve the identifier of the device with tag "Monitor A1"
        >>> device_tag = "Monitor A1"
        >>> device_id = sdk.get_device_id(device_tag)
        >>> print(device_id)
        1
        """
        if device_tag in self._device_ids:
            return self._device_ids[device_tag]
        row = self.sql_handler.select_device(device_tag=device_tag)
        if row is None:
            return None

        device_id = row[0]
        self._device_ids[device_tag] = device_id
        return device_id

    def get_device_info(self, device_id: int):
        """
        Retrieve information about a specific device in the linked relational database.

        :param int device_id: The identifier of the device to retrieve information for.

        :return: A dictionary containing information about the device, including its id, tag, name, manufacturer, model,
                 type, bed_id, and source_id.
        :rtype: dict

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> device_id = 1
        >>> device_info = sdk.get_device_info(device_id)
        >>> print(device_info)
        {'id': 1,
         'tag': 'Device A1',
         'name': 'Philips Device A1 in Room 1A',
         'manufacturer': 'Philips',
         'model': 'A1',
         'type': 'Device',
         'bed_id': 1,
         'source_id': 1}

        """

        if device_id in self._devices:
            return self._devices[device_id]
        row = self.sql_handler.select_device(device_id=device_id)

        if row is None:
            return None
        
        device_id, device_tag, device_name, device_manufacturer, device_model, device_type, device_bed_id, \
            device_source_id = row

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

        self._devices[device_id] = device_info

        return device_info

    def get_mrn_to_patient_id_map(self, mrn_list=None):
        patient_list = self.sql_handler.select_all_patients_in_list(mrn_list=mrn_list)
        return {row[1]: row[0] for row in patient_list}

    def get_patient_id_to_mrn_map(self, patient_id_list=None):
        patient_list = self.sql_handler.select_all_patients_in_list(patient_id_list=patient_id_list)
        return {row[0]: row[1] for row in patient_list}
