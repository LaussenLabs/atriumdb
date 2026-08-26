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
from collections import defaultdict
import numpy as np
import bisect

import threading
from contextlib import ExitStack

from atriumdb.intervals.intersection import intervals_intersect, list_intersection
from atriumdb.intervals.union import intervals_union_list
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.adb_functions import allowed_interval_index_modes, get_block_and_interval_data, condense_byte_read_list, \
    find_intervals, intervals_from_timestamps, intervals_from_gap_data, clip_intervals_to_source_ranges, \
    header_period_ns, is_string_dtype, is_fixed_width_string_dtype, same_write_value_kind, get_measure_period_ns, \
    require_measure_id, clip_patient_ranges, patient_sample_mask, append_block_cache, freeze_block_caches, \
    sort_data, collapse_duplicate_times, \
    yield_data, convert_to_nanoseconds, convert_to_nanohz, reconstruct_messages, \
    ALLOWED_TIME_TYPES, collect_all_descendant_ids, get_best_measure_id, _calc_end_time_from_gap_data, \
    merge_timestamp_data, merge_gap_data, create_timestamps_from_gap_data, freq_nhz_to_period_ns, time_unit_options, \
    create_gap_arr_from_variable_messages, sort_message_time_values, convert_from_nanoseconds, detect_period, \
    choose_interval_gap_tolerance, choose_time_encoding, collapse_continuous_write_intervals, \
    observed_median_delta_ns, observed_median_delta_from_gap_array, widen_gap_tolerance_for_observed_spacing, \
    APERIODIC_MIN_PERIOD_NS, DEFAULT_TIME_COMPRESSION_LEVEL, ENCODE_SAMPLE_SIZE, \
    INTERVAL_DENSITY_WARNING_RATIO, INTERVAL_DENSITY_WARNING_MIN_ROWS, DUPLICATE_KEEP_OPTIONS
from atriumdb.block import Block, create_gap_arr
from atriumdb.block_wrapper import T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, BlockMetadataWrapper
from atriumdb.file_api import AtriumFileHandler
from atriumdb.file_lock import make_file_lock
from atriumdb.string_dictionary import MeasureStringDictionary
from atriumdb.event_intervals import (
    union_windows, clip_spans_and_union, pair_from_to, clip_intervals_to_containers,
    collapse_event_intervals)

# The measure-metadata axes (signal_kind / value_type). These are stored as two
# independent nullable columns on the measure table; a NULL is read-time
# defaulted below so every existing dataset stays correct with no backfill.
# The vocabulary, the defaults and the predicates that go with them live in
# :mod:`atriumdb.measure_kinds` so the SDK, the windowing layer and the transfer
# layer describe the same model; re-exported here for backwards compatibility.
from atriumdb.measure_kinds import (
    DEFAULT_SIGNAL_KIND, DEFAULT_VALUE_TYPE,
    SIGNAL_KIND_WAVEFORM, VALUE_TYPE_NUMERIC, VALUE_TYPE_STRING, is_string_value_type,
    measure_kind_of, changed_kind_fields, is_invalid_kind_combination, validate_measure_kind_values,
    invalid_kind_combination_message, STRING_SIGNAL_KIND_FALLBACK)
from atriumdb.helpers import shared_lib_filename_windows, shared_lib_filename_macos, shared_lib_filename_linux, protected_mode_default_setting, \
    overwrite_default_setting
from atriumdb.helpers.settings import ALLOWABLE_OVERWRITE_SETTINGS, PROTECTED_MODE_SETTING_NAME, OVERWRITE_SETTING_NAME, \
    ALLOWABLE_PROTECTED_MODE_SETTINGS
from atriumdb.helpers.block_constants import TIME_TYPES_STR, VALUE_TYPES_STR, COMPRESSION_TYPES
from atriumdb.block_wrapper import BlockMetadata
import time
import atexit
from pathlib import Path, PurePath
import sys
import os
from typing import Union, List, Tuple, Optional
import platform
from ctypes import sizeof

from atriumdb.sql_handler.sql_constants import (
    SUPPORTED_DB_TYPES, MEASURE_ROW_ID, measure_row_signal_kind,
    measure_row_value_type,
)
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler
from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.filtered_iterator import FilteredDatasetIterator
from atriumdb.windowing.light_mapped_iterator import LightMappedIterator
from atriumdb.windowing.random_access_iterator import MappedIterator
from atriumdb.windowing.verify_definition import verify_definition
from atriumdb.windowing.windowing_functions import resolve_nominal_period_ns, validate_fill_rule_name
from atriumdb.windowing.definition_splitter import partition_dataset
from atriumdb.write_buffer import WriteBuffer

try:
    import requests
    from requests import Session
    from dotenv import dotenv_values, set_key
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

    **Where things are.** This class is large. Its methods are grouped into sections
    marked by ``# ---- <name> ----`` banners, in this source order:

    * *Construction and dataset creation* -- ``__init__``, :meth:`create_dataset`.
    * *Measure value-type state machine* -- private; how a measure becomes numeric or
      string-typed, and the guards that keep it that way.
    * *Measure metadata* -- :meth:`get_measure_info`, :meth:`update_measure`.
    * *Reading signal data* -- :meth:`get_data`, :meth:`get_string_data`.
    * *Event query surface* -- :meth:`get_measure_string_vocabulary`,
      :meth:`get_string_values_present`, :meth:`get_event_intervals`.
    * *Block-level and remote read paths* -- :meth:`get_data_from_blocks` and the
      API-mode read internals.
    * *Writing signal data (core)* -- :meth:`write_data_easy`, :meth:`write_data`.
    * *Writing signal data (entry points)* -- :meth:`write_buffer`, :meth:`write_segment`,
      :meth:`write_segments`, :meth:`write_time_value_pairs`.
    * *In-process caches* -- :meth:`load_device`, :meth:`load_definition`.
    * *Measures* -- :meth:`get_measure_id`, :meth:`get_measure_info`,
      :meth:`search_measures`, :meth:`insert_measure`.
    * *Devices* -- :meth:`get_device_id`, :meth:`get_device_info`, :meth:`insert_device`.
    * *Patients and patient history* -- :meth:`get_patient_id`, :meth:`get_patient_info`,
      :meth:`insert_patient`, :meth:`get_patient_history`.
    * *Device-patient mapping and encounters* -- :meth:`get_device_patient_mapping`,
      :meth:`insert_device_patient_data`, :meth:`get_encounters`, :meth:`insert_encounter`.
    * *Labels* -- :meth:`get_labels`, :meth:`insert_label`, :meth:`delete_labels`.
    * *Label names, sources and time series* -- :meth:`get_label_name_id`,
      :meth:`insert_label_name`, :meth:`get_label_time_series`.
    * *Windowed iteration* -- :meth:`get_iterator`.
    * *Interval / coverage queries* -- :meth:`get_interval_array`.
    * *Beds and sources* -- :meth:`get_bed_id`, :meth:`get_source_id` and their
      ``*_info`` siblings.
    * *API mode: HTTP transport, token refresh and shutdown* -- private request/token
      plumbing plus :meth:`close`.
    * *Low-level block/TSC file access, settings and remaining internals* --
      :meth:`get_data_from_tsc_file`, :meth:`get_blocks`,
      :meth:`get_batched_data_generator`.

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
    :param AtriumFileHandler storage_handler: Advanced feature. If you implement your own atriumdb file handler you can set it here.
    :param bool auto_upgrade: If True, automatically upgrade the database schema if needed (e.g., adding new columns). This allows the SDK to initialize successfully even if the database schema is outdated. Default is False.

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

    # ------------------------------------------------------------------ #
    # Construction and dataset creation
    # ------------------------------------------------------------------ #
    def __init__(self, dataset_location: Union[str, PurePath] = None, metadata_connection_type: str = None,
                 connection_params: dict = None, num_threads: int = 1, api_url: str = None, token: str = None,
                 refresh_token=None, validate_token=True, tsc_file_location: str = None, atriumdb_lib_path: str = None,
                 no_pool=False, storage_handler: AtriumFileHandler = None, auto_upgrade: bool = False):
        self.block_cache = {}  # device_id -> list of block sql results
        self.start_cache = {}  # device_id -> list of start times
        self.end_cache = {}  # device_id -> list of end times
        self.filename_dict = {}

        self.label_cache = {}  # device_id -> label_name_id -> list of label records
        self.label_start_cache = {}  # device_id -> label_name_id -> list of start times
        self.label_end_cache = {}  # device_id -> label_name_id -> list of end times
        self.descendant_cache = {}  # label_name_id -> (all_descendants_set, closest_ancestor_dict)
        self.label_lookup_caches = {}

        self.dataset_location = dataset_location

        # Set default metadata connection type if not provided
        metadata_connection_type = DEFAULT_META_CONNECTION_TYPE if \
            metadata_connection_type is None else metadata_connection_type

        self.metadata_connection_type = metadata_connection_type

        # Set the C DLL path based on the platform if not provided
        if atriumdb_lib_path is None:
            if sys.platform == "win32":
                shared_lib_filename = shared_lib_filename_windows
            elif platform.system() == "Darwin":
                # macOS requires a locally built dylib; see tsc-lib/build_mac.sh.
                shared_lib_filename = shared_lib_filename_macos
            else:
                shared_lib_filename = shared_lib_filename_linux

            this_file_path = Path(__file__)
            atriumdb_lib_path = this_file_path.parent.parent / shared_lib_filename

            if platform.system() == "Darwin" and not Path(atriumdb_lib_path).exists():
                raise OSError("AtriumSDK on macOS requires a locally built codec library: "
                              f"{atriumdb_lib_path} not found. Build it with tsc-lib/build_mac.sh "
                              "(requires cmake, libomp, lz4 and zstd from Homebrew).")

        # Initialize the block object with the C DLL path and number of threads
        self.block = Block(atriumdb_lib_path, num_threads)

        # Initialize write buffer param
        self._active_buffer = None

        # Setup SQL Handler
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
            if not db_file.exists():
                raise ValueError(f"No Dataset found at location {dataset_location}. "
                                 f"Use AtriumSDK.create_dataset to create a new dataset.")
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Initialize the SQLiteHandler with the database file path
            self.sql_handler = SQLiteHandler(db_file)
            self._init_local_mode(tsc_file_location, storage_handler, auto_upgrade, "TEXT")


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

            # Initialize the MariaDBHandler with the connection parameters
            from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler, maria_connection_args
            self.sql_handler = MariaDBHandler(*maria_connection_args(connection_params), no_pool=no_pool)
            self._init_local_mode(tsc_file_location, storage_handler, auto_upgrade, "TEXT/VARCHAR")


        # Handle API connections
        elif metadata_connection_type == 'api':
            # Check if the necessary modules are installed for API connections
            if not REQUESTS_INSTALLED:
                raise ImportError("Remote mode not installed. Please install atriumdb with pip install atriumdb[remote]")

            self.file_api = storage_handler if storage_handler else AtriumFileHandler(None)

            self.mode = "api"
            self.api_url = api_url
            self.validate_token = validate_token
            # remove the leading http stuff and replace it with ws, also remove any trailing slashes
            self.ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://").rstrip('/')
            # make this variable so once connection is made in the thread it is available to the sdk object
            self.websock_conn = None

            # need this variable so when we refresh the token we know if a .env file was supplied and we should set the token key
            self.dot_env_loaded = False

            # Load API and refresh token from ./.env or the environment if not
            # provided. The .env values are read directly (a value in the file
            # wins over the inherited environment) rather than loaded into
            # os.environ, so constructing an SDK never mutates the process
            # environment.
            if token is None or refresh_token is None:
                dot_env_values = dotenv_values(dotenv_path="./.env")
                if token is None:
                    token = dot_env_values.get('ATRIUMDB_API_TOKEN') or os.environ.get('ATRIUMDB_API_TOKEN')
                    self.dot_env_loaded = token is not None
                if refresh_token is None:
                    refresh_token = dot_env_values.get('ATRIUMDB_API_REFRESH_TOKEN') \
                        or os.environ.get('ATRIUMDB_API_REFRESH_TOKEN')

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
        self._label_source_ids, self._label_sources = {}, {}

        # Initialize measures and devices if not in API mode
        if metadata_connection_type != "api":
            self._measures = self.get_all_measures()
            self._devices = self.get_all_devices()
            self._label_sets = self.get_all_label_names()

            self._label_sources = self.get_all_label_sources()
            self._label_source_ids = {}

            for source_id, source_info in self._label_sources.items():
                self._label_source_ids[source_info['name']] = source_id

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

    def _init_local_mode(self, tsc_file_location, storage_handler, auto_upgrade, mrn_column_type: str):
        """Finish constructing a local-mode SDK once ``self.sql_handler`` exists.

        Identical for both metadata backends apart from ``mrn_column_type``, which
        only names the required column type in the "please upgrade" error."""
        self.mode = "local"
        self.file_api = storage_handler if storage_handler else AtriumFileHandler(tsc_file_location)

        dataset_version = self.sql_handler.get_dataset_schema_version()
        if dataset_version is not None and dataset_version > self.sql_handler.CURRENT_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"This dataset was written by a newer AtriumDB (schema version {dataset_version}) than this "
                f"installed version understands (schema version {self.sql_handler.CURRENT_DATASET_SCHEMA_VERSION}). "
                f"Downgrading a dataset is not supported; install a newer atriumdb package instead of opening it "
                f"with this one."
            )

        if auto_upgrade:
            self.sql_handler.update_measure_schema()
            self.sql_handler.upgrade_mrn_schema()
            for measure_id, tag, period_ns in self.sql_handler.repair_zero_freq_measures():
                _LOGGER.info(
                    "Converted legacy aperiodic measure %d (%s) from freq_nhz=0 to "
                    "signal_kind='sample' with a nominal period of %d ns, observed from "
                    "its own blocks.", measure_id, tag, period_ns)
            self.sql_handler.ensure_interval_union_procedure()
            self._backfill_string_value_types()
            self.sql_handler.record_dataset_schema_version()
        elif not self.sql_handler.check_mrn_column_is_text():
            raise ValueError(
                f"The 'mrn' column in the patient table is using an INTEGER type, but {mrn_column_type} is now "
                f"required. Please run AtriumSDK(auto_upgrade=True) to update the database schema."
            )
        else:
            pending = self.sql_handler.pending_schema_upgrades()
            if pending:
                _LOGGER.warning(
                    "This dataset is missing %s. It will work, using a fallback, but "
                    "AtriumSDK(auto_upgrade=True) will bring it up to date.",
                    " and ".join(pending))

        self.settings_dict = self._get_all_settings()

    @classmethod
    def create_dataset(cls, dataset_location: Union[str, PurePath], database_type: str = None, protected_mode: str = None,
                       overwrite: str = None, connection_params: dict = None, no_pool=False, auto_upgrade: bool = False):
        """
        .. _create_dataset_label:

        A class method to create a new dataset.

        :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the dataset will be written.
        :param str database_type: Specifies the type of metadata database to use. Options are "sqlite", "mysql", or "mariadb".
        :param str protected_mode: Specifies the protection mode of the metadata database. Allowed values are "True" or "False". If "True", data deletion will not be allowed. If "False", data deletion will be allowed. The default behavior can be changed in the `sdk/atriumdb/helpers/config.toml` file.
        :param str overwrite: The dataset's merge conflict policy: how block merging resolves duplicate timestamps
            between a new write and existing data. "overwrite" and the legacy default "ignore" keep the new write's
            values; "protect" keeps the existing values; "error" raises when a write would conflict with existing
            data. The policy is enforced where deduplication happens - writes smaller than one block that merge with
            an existing block, and duplicate pushes within one buffer flush. Overlapping writes of a full block or
            more never merge, so both copies are stored regardless of this setting -- that is accepted (write speed
            is the priority) and is resolved on READ with ``get_data(..., allow_duplicates=False)``, which keeps one
            sample per timestamp using this same policy: the newer write's value under "overwrite"/"ignore", the
            existing value under "protect".
        :param dict connection_params: A dictionary containing connection parameters for "mysql" or "mariadb" database type. It should contain keys for 'host', 'user', 'password', 'database', and 'port'.
        :param bool no_pool: If true disables Mariadb connection pooling, instead using a new connection for each query.
        :param bool auto_upgrade: If True, automatically upgrade the database schema if needed (e.g., adding new columns). This allows the SDK to initialize successfully even if the database schema is outdated. Default is False.

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
            from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler, maria_connection_args
            MariaDBHandler(*maria_connection_args(connection_params)).create_schema()

        sdk_object = cls(dataset_location=dataset_location, metadata_connection_type=database_type,
                         connection_params=connection_params, no_pool=no_pool, auto_upgrade=auto_upgrade)

        # Add settings
        sdk_object.sql_handler.insert_setting(PROTECTED_MODE_SETTING_NAME, str(protected_mode))
        sdk_object.sql_handler.insert_setting(OVERWRITE_SETTING_NAME, str(overwrite))

        sdk_object.settings_dict = sdk_object._get_all_settings()

        return sdk_object

    # ------------------------------------------------------------------ #
    # Measure value-type state machine (internal): establish, resolve, guard
    # ------------------------------------------------------------------ #
    @property
    def _meta_dir(self) -> Path:
        """The dataset's ``meta/`` directory, derived from ``dataset_location``.

        ``meta/`` already holds the SQLite index and transfer's ``definition.yaml``;
        per-measure string dictionaries live under ``meta/string_dict/``. Requires a
        ``dataset_location`` (always present in local/sqlite/mariadb modes; string
        storage is not supported in pure API mode)."""
        if self.dataset_location is None:
            raise ValueError(
                "String value support requires a dataset_location (the meta/ directory); "
                "it is unavailable in this SDK mode.")
        return Path(self.dataset_location) / "meta"

    def _dictionary_establishes_string(self, measure_id):
        """The single rule for "this measure's dictionary file proves it is string-typed".

        A dictionary file alone is NOT proof. ``write_data`` appends to the dictionary
        *before* the block bytes and SQL rows are committed (the codes have to be baked
        into the encoded block), so a write killed in between -- SIGKILL, power loss, or a
        rollback that could not run because a concurrent appender owned the tail -- leaves
        a dictionary describing data the dataset does not contain. Treating that husk as an
        establishment would lock a numeric measure to 'string' with no public way back.
        Requiring a committed block makes the husk inert and self-healing: the measure
        stays unestablished, and whichever kind of data commits first establishes it.

        Every consumer of the dictionary-file signal (:meth:`_resolve_measure_kind`,
        :meth:`_established_value_type`, :meth:`_backfill_string_value_types`) must use
        this one rule, or they disagree with each other -- e.g. ``get_measure_info``
        serving 'string' off a bare file while the write path resolves 'None' off the
        same measure and happily accepts numeric data.

        The cheap filesystem check is deliberately evaluated first so the block query only
        runs for a measure that has a dictionary file AND a NULL value_type column -- after
        a measure's first committed write the column is set and neither check is reached.
        """
        if measure_id is None or self.dataset_location is None:
            return False
        if not MeasureStringDictionary.exists(self._meta_dir, measure_id):
            return False
        return bool(self.sql_handler.measure_has_blocks(measure_id))

    def _backfill_string_value_types(self):
        """Opportunistic, idempotent backfill: any measure that already has a
        string-dictionary file *with committed blocks behind it* is marked
        ``value_type='string'`` in the measure table. Read-time defaults make this
        unnecessary for correctness (a NULL value_type with a dict file still reads
        as string); persisting it lets later reads skip the filesystem check. Safe
        to re-run; never overwrites an already-set value_type.

        The block requirement is load-bearing, not an optimization: without it this
        backfill *persists* a poisoning value_type. An orphan dictionary left by a killed
        write would make the very next ``AtriumSDK(auto_upgrade=True)`` write
        ``value_type='string'`` into the column, which then permanently rejects the
        numeric writes the measure actually exists for -- a self-inflicted brick on a
        routine schema upgrade."""
        if self.dataset_location is None:
            return
        for row in self.sql_handler.select_all_measures():
            measure_id, stored_value_type = row[MEASURE_ROW_ID], measure_row_value_type(row)
            if stored_value_type is None and self._dictionary_establishes_string(measure_id):
                self.sql_handler.update_measure_metadata(measure_id, value_type=VALUE_TYPE_STRING)

    def _resolve_measure_kind(self, measure_id, stored_signal_kind, stored_value_type):
        """Apply the read-time defaults to the raw measure columns.

        ``NULL signal_kind`` -> ``waveform``. ``NULL value_type`` -> ``string``
        when a string-dictionary file establishes it (see
        :meth:`_dictionary_establishes_string` -- a dictionary with committed blocks
        behind it, so un-migrated / un-backfilled datasets still read correctly),
        otherwise ``numeric``. The column, when set, always wins. This is the single
        source of truth used by ``get_measure_info`` / ``get_all_measures`` and the
        ``get_data`` guard."""
        signal_kind = stored_signal_kind if stored_signal_kind is not None else DEFAULT_SIGNAL_KIND
        if stored_value_type is not None:
            value_type = stored_value_type
        elif self._dictionary_establishes_string(measure_id):
            value_type = VALUE_TYPE_STRING
        else:
            value_type = DEFAULT_VALUE_TYPE
        return signal_kind, value_type

    def _established_value_type(self, measure_id):
        """Return the measure's already-established value_type ('string'/'numeric')
        or None if the measure has no data yet (first write may establish it).

        Distinct from :meth:`_resolve_measure_kind`, which read-time-defaults a
        brand-new measure to ``numeric``; here an un-written measure returns
        ``None`` so a first string write is not wrongly rejected. Resolution order:
        the raw ``value_type`` column, then a dictionary file *with committed
        blocks* (-> string), then the presence of any block (-> numeric).

        The dictionary file only establishes ``string`` when the measure also has
        blocks: a dictionary with no data behind it is the fingerprint of a write
        that was rolled back or killed mid-flight, and honouring it would lock a
        numeric measure to ``string`` forever with no public way back."""
        row = self.sql_handler.select_measure(measure_id=measure_id)
        if row is None:
            return None
        stored_value_type = measure_row_value_type(row)
        if stored_value_type is not None:
            return stored_value_type
        # Same rule as _resolve_measure_kind / the backfill, so the three can never
        # disagree about the same measure.
        if self._dictionary_establishes_string(measure_id):
            return VALUE_TYPE_STRING
        if self.sql_handler.measure_has_blocks(measure_id):
            return VALUE_TYPE_NUMERIC
        return None

    def _check_value_type_invariant(self, measure_id, incoming_is_string):
        """Raise if this write's value-kind conflicts with the measure's already
        established value_type. Does NOT persist anything -- establishment happens
        only *after* the write passes its own validation and commits (see
        :meth:`_establish_value_type`), so a write that raises downstream can never
        leave a poisoning value_type behind. Returns the established value_type, or
        None if the measure has no data yet."""
        incoming = VALUE_TYPE_STRING if incoming_is_string else VALUE_TYPE_NUMERIC
        established = self._established_value_type(measure_id)
        if established is not None and established != incoming:
            raise ValueError(
                f"Measure {measure_id} is a '{established}' measure; cannot write "
                f"'{incoming}' values to it. A measure is either numeric or string -- "
                f"mixing the two on one measure corrupts readability. Write "
                f"'{incoming}' data to a separate measure.")
        return established

    def _establish_value_type(self, measure_id, incoming_is_string):
        """Persist the measure's value_type after a successful write, iff the
        column is not already set. Called only once the write has committed, so a
        write that raises can never establish (and thus poison) a value_type. The
        prior :meth:`_check_value_type_invariant` already guaranteed no conflict.

        The in-process measure cache is dropped either way, so a cached view can
        never disagree with the persisted row -- a stale cache saying 'numeric' while
        the write path resolves 'string' leaves the measure rejecting numeric writes
        AND string reads."""
        row = self.sql_handler.select_measure(measure_id=measure_id)
        if row is None:
            return
        stored_value_type = measure_row_value_type(row)
        if stored_value_type is not None:
            # Already established (possibly by another process). Drop a cached view
            # that disagrees rather than serving a stale value_type.
            cached = self._measures.get(measure_id)
            if cached is not None and cached.get('value_type') != stored_value_type:
                self._measures.pop(measure_id, None)
            return
        # A first string write establishes value_type='string'. If the measure's
        # signal_kind is (or read-time-defaults to) 'waveform' that lands the measure in
        # the one combination the design forbids -- and this is the exact route the docs'
        # former string example took: insert_measure() with no signal_kind, then
        # write_time_value_pairs() with text. get_string_data() then works and
        # get_iterator() dies hours later. Repair the shape in the same statement that
        # establishes the encoding, and say so loudly.
        repaired_signal_kind = None
        if incoming_is_string and is_invalid_kind_combination(
                measure_row_signal_kind(row), VALUE_TYPE_STRING):
            repaired_signal_kind = STRING_SIGNAL_KIND_FALLBACK
            _LOGGER.warning(invalid_kind_combination_message(measure_id, repaired_signal_kind))

        self.sql_handler.update_measure_metadata(
            measure_id,
            signal_kind=repaired_signal_kind,
            value_type=(VALUE_TYPE_STRING if incoming_is_string else VALUE_TYPE_NUMERIC))
        self._measures.pop(measure_id, None)

    def _block_merge_lock(self, measure_id, device_id):
        """Exclusive cross-process lock for the small-write block merge of one
        (measure, device) stream.

        ``write_data``'s merge path is a read-modify-write: select the closest block,
        decode its file, merge, write a new file, delete the old block row. Nothing
        isolated those steps, so concurrent writers to the same stream silently lost
        writes -- and events, which are always far below ``block_size``, take that path on
        every single write. This lock makes the sequence atomic between processes and
        threads alike.

        Keyed per (measure, device), because that is the granularity the merge actually
        conflicts at, so independent streams keep writing in parallel. Held only around
        the merge itself; ordinary block-sized writes never take it.

        Advisory and filesystem-based (see :mod:`atriumdb.file_lock`): it coordinates
        writers that share a ``dataset_location``. Without one -- the storage-handler /
        API configurations, which do not merge locally -- there is nothing to key on and
        this degrades to a no-op rather than pretending to protect anything.

        The zero-byte lock files are nested one directory per measure so no single
        directory ever holds measures x devices entries."""
        if self.dataset_location is None:
            return ExitStack()  # inert context manager
        return make_file_lock(
            self._meta_dir / "locks" / f"measure_{int(measure_id)}" / f"device_{int(device_id)}.lock")

    def _check_string_dictionary_not_lost(self, measure_id, string_dict, established_value_type,
                                          watermark):
        """Refuse a string write whose dictionary has fewer entries than the codes
        already committed to this measure's blocks.

        The dictionary is a file under ``meta/`` and the blocks that reference its codes
        are indexed in the metadata database; the two are restored independently, so a
        DB + ``tsc/`` restore that omits ``meta/`` leaves the codes with no vocabulary.
        The next write would then start again at code 0 and every historical code would
        silently begin decoding to a DIFFERENT string -- no error, no warning, permanently
        wrong clinical values. This is the guard that makes that impossible.

        Two O(1) checks, both indexed lookups -- no block is read and no data is decoded:

        1. The high-water mark recorded in the metadata database (which survives the loss
           of ``meta/``) is the vocabulary size of the last committed string write. A
           dictionary file shorter than it has lost entries. Catches total loss *and* tail
           truncation. ``watermark`` must have been read BEFORE ``string_dict`` was loaded
           (see the call site) or a concurrent writer's commit turns this into a false
           positive.
        2. Datasets written before the mark existed have none, so fall back to the fact
           that a measure the database has *established* as string, with blocks behind it,
           cannot legitimately have an empty vocabulary -- every committed string write
           leaves at least one entry. Catches total loss on legacy datasets.

        Neither check can fire for a first write to a fresh measure: there is no mark, and
        an unwritten measure is not established.
        """
        vocabulary_size = len(string_dict)
        remedy = (
            "Restore meta/string_dict/ from the backup taken with this dataset's .tsc files "
            "and metadata database, and do not write to this measure until you do -- writing "
            "now would re-issue codes that committed blocks already use, silently changing "
            "what historical values decode to. If the dictionary is genuinely unrecoverable, "
            "the measure's existing string data cannot be read back and the measure should be "
            "retired rather than appended to.")

        if watermark is not None and vocabulary_size < watermark:
            raise ValueError(
                f"String dictionary for measure {measure_id} has lost data: "
                f"'{MeasureStringDictionary.path_for(self._meta_dir, measure_id)}' holds "
                f"{vocabulary_size} entries but {watermark} were committed (blocks for this "
                f"measure may reference codes up to {watermark - 1}). " + remedy)

        if watermark is None and vocabulary_size == 0 \
                and is_string_value_type(established_value_type) \
                and self.sql_handler.measure_has_blocks(measure_id):
            raise ValueError(
                f"String dictionary for measure {measure_id} is missing or empty "
                f"('{MeasureStringDictionary.path_for(self._meta_dir, measure_id)}'), but the "
                f"measure is recorded as a string measure and already holds committed blocks, "
                f"so those blocks reference codes this dictionary can no longer resolve. "
                + remedy)

    def _record_string_dictionary_size(self, measure_id, vocabulary_size):
        """Persist the vocabulary size a just-committed string write may reference.

        Best-effort and non-fatal: the blocks are already durable at this point, so
        failing the caller's write because the watermark could not be recorded would turn
        a successful write into a spurious error. A missed update only weakens the
        dictionary-loss guard to total-loss detection for this measure."""
        try:
            self.sql_handler.set_string_dict_watermark(measure_id, vocabulary_size)
        except Exception as watermark_error:  # pragma: no cover - defensive
            _LOGGER.warning(
                f"Could not record the string dictionary size for measure {measure_id} "
                f"({vocabulary_size} entries). The dictionary-loss guard for this measure "
                f"falls back to detecting total loss only: {watermark_error}")

    def _apply_kind_to_existing_measure(self, measure_id, signal_kind, value_type):
        """Apply explicitly-requested kind metadata on ``insert_measure``'s
        get-or-insert path instead of silently discarding it.

        ``insert_measure`` returns the existing id when tag/freq/units already match.
        Dropping ``signal_kind`` there would mean an ingest pipeline that creates its
        measures with get-or-insert could never classify them: the measure would stay
        ``waveform``, and ``waveform`` + ``string`` is the combination the windowing
        layer cannot iterate.

        Only stated fields are applied, and only when they actually differ (so the
        common repeat-insert stays a no-op). A ``value_type`` that conflicts with data
        already written raises, exactly as a conflicting write would."""
        if signal_kind is None and value_type is None:
            return
        current_info = self.get_measure_info(measure_id)
        if current_info is None:
            return
        current_signal_kind, current_value_type = measure_kind_of(current_info)
        new_signal_kind, new_value_type = changed_kind_fields(
            current_signal_kind, current_value_type, signal_kind, value_type)
        if new_signal_kind is None and new_value_type is None:
            return
        _LOGGER.warning(
            f"insert_measure: measure {measure_id} already exists; applying the requested "
            f"metadata to it (signal_kind {current_signal_kind!r} -> {new_signal_kind or current_signal_kind!r}, "
            f"value_type {current_value_type!r} -> {new_value_type or current_value_type!r}). "
            f"Use update_measure() to change this explicitly.")
        self.update_measure(measure_id, signal_kind=new_signal_kind, value_type=new_value_type)

    # ------------------------------------------------------------------ #
    # Measure metadata
    # ------------------------------------------------------------------ #
    def update_measure(self, measure_id: int, *, signal_kind: str = None, value_type: str = None):
        """Set (or correct) a measure's ``signal_kind`` / ``value_type`` after it
        was created. It intentionally groups mutable metadata under one update
        operation instead of adding a setter for each measure property.

        ``insert_measure`` is a get-or-insert, so a measure auto-created by an ingest
        pipeline (or by an earlier transfer) commonly ends up with the default
        ``waveform`` shape. ``value_type`` self-heals on the first write, but
        ``signal_kind`` never does, and ``waveform`` + ``string`` is precisely the
        combination the windowing layer cannot iterate. This update operation is the
        public way to correct both.

        :param int measure_id: The measure to update.
        :param str signal_kind: New temporal shape, one of ``waveform | sample | event |
            state``. ``None`` leaves it unchanged. Purely descriptive metadata -- safe to
            change at any time.
        :param str value_type: New value encoding, one of ``numeric | string``. ``None``
            leaves it unchanged. Changing it to conflict with data that has already been
            written raises ``ValueError``: the stored blocks are either dictionary codes
            or numbers and re-labelling them would make the measure unreadable.

        :return: The updated measure record, including resolved metadata.
        :rtype: dict

        >>> measure_id = sdk.insert_measure("vent_mode", freq=1, freq_units="Hz", units="string")
        >>> sdk.update_measure(measure_id, signal_kind="state", value_type="string")['signal_kind']
        'state'
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for measure updates.")
        validate_measure_kind_values(signal_kind, value_type)

        if self.sql_handler.select_measure(measure_id=int(measure_id)) is None:
            raise ValueError(f"measure_id {measure_id} not found in the dataset.")

        if value_type is not None:
            established = self._established_value_type(int(measure_id))
            if established is not None and established != value_type:
                raise ValueError(
                    f"Measure {measure_id} already holds '{established}' data; it cannot be "
                    f"relabelled as '{value_type}'. A measure is either numeric or string -- "
                    f"write '{value_type}' data to a separate measure.")

        current_info = self.get_measure_info(int(measure_id))
        current_kind = measure_kind_of(current_info) if current_info else (None, None)
        resulting_signal_kind = signal_kind if signal_kind is not None else current_kind[0]
        resulting_value_type = value_type if value_type is not None else current_kind[1]
        if is_invalid_kind_combination(resulting_signal_kind, resulting_value_type):
            _LOGGER.warning(invalid_kind_combination_message(measure_id, STRING_SIGNAL_KIND_FALLBACK))
            signal_kind = STRING_SIGNAL_KIND_FALLBACK

        if signal_kind is not None or value_type is not None:
            self.sql_handler.update_measure_metadata(
                int(measure_id), signal_kind=signal_kind, value_type=value_type)
            # Never serve a cached view that predates the change.
            self._measures.pop(int(measure_id), None)

        return self.get_measure_info(int(measure_id))

    # ------------------------------------------------------------------ #
    # Reading signal data: numeric and string
    # ------------------------------------------------------------------ #
    def _require_measure_and_device(self, measure_id, device_id):
        """The measure and device rows for a write, or a ValueError naming whichever
        one is missing and the call that creates it."""
        measure_info = self.get_measure_info(measure_id)
        device_info = self.get_device_info(device_id)

        if measure_info is None:
            raise ValueError(f"measure_id {measure_id} not found in the dataset. "
                             f"Add it with AtriumSDK.insert_measure(tag, freq, units)")
        if device_info is None:
            raise ValueError(f"device_id {device_id} not found in the dataset. "
                             f"Add it with AtriumSDK.insert_device(tag)")

        return measure_info, device_info

    def _resolve_read_range(self, start_time_n, end_time_n, time_units, device_id, device_tag,
                            patient_id, mrn):
        """Front half of a read query, shared by get_data and get_headers: validate
        ``time_units``, convert the range into nanoseconds, and resolve
        ``device_tag`` / ``mrn`` to ids.

        Returns ``(time_units, start_time_n, end_time_n, device_id, patient_id)``."""
        time_units = "ns" if time_units is None else time_units

        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        start_time_n = int(start_time_n * time_unit_options[time_units])
        end_time_n = int(end_time_n * time_unit_options[time_units])

        if device_id is None and device_tag is not None:
            device_id = self.get_device_id(device_tag)

        if patient_id is None and mrn is not None:
            patient_id = self.get_patient_id(mrn)

        return time_units, start_time_n, end_time_n, device_id, patient_id

    def _resolve_read_measure_id(self, measure_id, measure_tag, freq, units, freq_units):
        """Resolve the signal half of a read query. API mode requires the full
        (tag, freq, units) triple; local mode picks the best match from the tag and
        coerces the result to an int."""
        if self.mode == "api":
            if measure_id is None:
                assert measure_tag is not None and freq is not None and units is not None, \
                    "Must provide measure_id or all of measure_tag, freq, units"
                measure_id = self.get_measure_id(measure_tag, freq, units, freq_units)
            return measure_id

        if measure_id is None:
            assert measure_tag is not None, "One of measure_id, measure_tag must be specified."
            measure_id = get_best_measure_id(self, measure_tag, freq, units, freq_units)

        return int(measure_id) if measure_id is not None else measure_id

    def _select_read_blocks(self, measure_id, device_id, patient_id, start_time_n, end_time_n,
                            block_info):
        """Resolve the blocks a read must touch, from whichever of the three sources
        applies: the in-memory block cache, a caller-supplied ``block_info``, or the
        metadata table.

        Returns ``(block_list, filename_dict, source)`` where ``source`` is one of
        ``"cache" | "block_info" | "db"``. An empty ``block_list`` means the range
        holds no data; callers decide what that means for their own return type --
        which is why ``source`` comes back, since only the two non-``block_info``
        sources nan-fill an empty numeric read."""
        if device_id is not None and measure_id is not None and \
                measure_id in self.block_cache and device_id in self.block_cache[measure_id]:
            return self.find_blocks(measure_id, device_id, start_time_n, end_time_n), \
                self.filename_dict, "cache"

        if block_info is not None:
            return block_info['block_list'], block_info['filename_dict'], "block_info"

        block_list = self.sql_handler.select_blocks(
            int(measure_id), int(start_time_n), int(end_time_n), device_id, patient_id
        )

        read_list = condense_byte_read_list(block_list)
        if len(read_list) == 0:
            return [], {}, "db"

        file_id_list = [row[2] for row in read_list]
        return block_list, self.get_filename_dict(file_id_list), "db"

    def get_data(self, measure_id: int = None, start_time_n: int = None, end_time_n: int = None,
                 device_id: int = None, patient_id=None, time_type=1, analog=True, block_info=None,
                 time_units: str = None, sort=True, allow_duplicates=True, measure_tag: str = None,
                 freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                 device_tag: str = None, mrn: str = None, return_nan_filled: bool | np.ndarray = False,
                 duplicate_keep: str = None):
        """
        .. _get_data_label:

        The method for querying data from the dataset, indexed by signal type (measure_id or measure_tag with freq and units),
        time (start_time_n and end_time_n), and data source (device_id, device_tag, patient_id, or mrn).

        If measure_id is None, measure_tag along with freq and units must not be None, and vice versa.
        Similarly, if device_id is None, device_tag must not be None, and if patient_id is None, mrn must not be None.

        :param int measure_id: The measure identifier. If None, measure_tag must be provided.
        :param int start_time_n: The start epoch in nanoseconds of the data you would like to query.
        :param int end_time_n: The end epoch in nanoseconds. The end time is not inclusive.
        :param int device_id: The device identifier. If None, device_tag must be provided.
        :param int patient_id: The patient identifier. If None, mrn must be provided.
        :param time_type: The type of time returned. Options are:
            - 1: Timestamps (default).
            - 2: Gap array (advanced users only).
            Patient-scoped reads require timestamp output (``time_type=1``), so the
            decoder can enforce each device-patient encounter boundary.
            - 'raw': Return as was originally stored.
            - 'encoded': Return in the format currently encoded (usually 2 for periodic signals).
        :param bool analog: Convert numeric values to analog signal. For a string
            measure, the default ``True`` returns decoded strings; ``False`` returns
            raw dictionary codes for advanced callers.
        :param block_info: Custom block_info list to skip metadata table check.
        :param str time_units: Unit for the time array returned. Options: ["s", "ms", "us", "ns"].
        :param bool sort: Whether to sort the returned data by time. Sorting is only applied when time_type is 1.
        :param bool allow_duplicates: Whether duplicate timestamps may appear in the returned data.
            **Default True, which returns every stored sample** -- unchanged behaviour.

            Set it to False to collapse duplicates on read. This is the supported way to deal
            with duplicate samples: the write path deduplicates only as a *side effect* of the
            small-write block merge (write speed is the priority, and a live feed that replays a
            large buffer legitimately stores the same timestamps twice), so surviving duplicates
            are resolved here instead.

            Semantics: a **duplicate is two samples with the same timestamp**, decided on the
            timestamp alone -- the same thing ``write_data``'s merge means by it -- whether or not
            the two carry the same value. Exactly one sample per timestamp is returned. Which one
            survives follows this dataset's ``overwrite`` merge conflict policy, so a read resolves
            a duplicate the way a write would have: under ``'overwrite'`` / ``'ignore'`` (the
            default) the most recently written copy wins; under ``'protect'`` the earliest written
            copy wins. ``duplicate_keep`` overrides that per call.

            Only applies when ``sort=True`` and ``time_type=1`` (collapsing requires ordering).
            It is vectorized -- one stable sort and one mask, no per-sample loop -- but it does
            cost that sort, so it is off by default.
        :param str measure_tag: A short string identifying the signal. Required if measure_id is None.
        :param freq: The sample frequency of the signal. Helpful with measure_tag.
        :param str units: The units of the signal. Helpful with measure_tag.
        :param str freq_units: Units for frequency. Options: ["nHz", "uHz", "mHz", "Hz", "kHz", "MHz"] default "nHz".
        :param str device_tag: A string identifying the device. Exclusive with device_id.
        :param str mrn: Medical record number for the patient. Exclusive with patient_id. An int can be provided, but will be converted and stored as a string.
        :param bool | ndarray return_nan_filled: Whether or not to fill missing values from start to end with np.nan.
            It is not available for string measures.
            It is also unavailable for patient-scoped reads, whose data can span
            multiple device-specific encounter windows.
            This can be floating point numpy array of shape (int(round((end_ns - start_ns) / period_ns),) which works
            like the `out` param in the numpy library, filling the result into the passed in array instead of creating
            a new array, which provides a modest performance increase if you already have a result array allocated.
        :param str duplicate_keep: Which copy of a duplicated timestamp survives when
            ``allow_duplicates=False``: ``"last"`` (the most recently written) or ``"first"`` (the
            earliest written). Default None follows the dataset's ``overwrite`` merge conflict
            policy as described above. Ignored when ``allow_duplicates`` is True.

        :rtype: Tuple[List[BlockMetadata], numpy.ndarray, numpy.ndarray]
        :returns: List of Block header objects, 1D numpy array for time data, 1D numpy array for value data.

        Examples:

        >>> # Every stored sample, duplicates included (default, unchanged).
        >>> headers, times, values = sdk.get_data(measure_id, start, end, device_id=device_id)
        >>> # One sample per timestamp; the most recently written value wins.
        >>> headers, times, values = sdk.get_data(
        ...     measure_id, start, end, device_id=device_id, allow_duplicates=False)
        """
        # check that a correct unit type was entered
        # make sure time type is either 1 or 2
        if time_type not in ALLOWED_TIME_TYPES:
            raise ValueError("Time type must be in [1, 2]")

        time_units, start_time_n, end_time_n, device_id, patient_id = self._resolve_read_range(
            start_time_n, end_time_n, time_units, device_id, device_tag, patient_id, mrn)

        measure_id = self._resolve_read_measure_id(measure_id, measure_tag, freq, units, freq_units)

        # If the data is from the api.
        if self.mode == "api":
            return self._get_data_api(measure_id, start_time_n, end_time_n, device_id=device_id, patient_id=patient_id,
                                      time_type=time_type, analog=analog, sort=sort, allow_duplicates=allow_duplicates)

        device_id = int(device_id) if device_id is not None else device_id

        is_string_measure = False
        if measure_id is not None and self.dataset_location is not None:
            _minfo = self.get_measure_info(measure_id)
            is_string_measure = _minfo is not None and is_string_value_type(_minfo.get('value_type'))
        if is_string_measure:
            if isinstance(return_nan_filled, np.ndarray) or return_nan_filled:
                raise ValueError(
                    f"Measure {measure_id} is a string measure; its values cannot be NaN-filled. "
                    f"Use AtriumSDK.get_string_data(...) to read string data.")
        decode_string_values = is_string_measure and analog
        string_dict = None
        if decode_string_values:
            # Dictionary codes are numeric storage detail, not analog values.
            string_dict = MeasureStringDictionary.load(self._meta_dir, int(measure_id))
            self._check_string_dictionary_not_lost(
                int(measure_id), string_dict, self._established_value_type(int(measure_id)),
                self.sql_handler.get_string_dict_watermark(int(measure_id)))
            analog = False

        block_list, filename_dict, block_source = self._select_read_blocks(
            measure_id, device_id, patient_id, start_time_n, end_time_n, block_info)

        if patient_id is not None and time_type != 1:
            raise ValueError(
                "patient-scoped reads require time_type=1 so each sample can be "
                "clipped to its device-patient encounter.")
        if patient_id is not None and (isinstance(return_nan_filled, np.ndarray) or return_nan_filled):
            raise ValueError(
                "patient-scoped reads do not support return_nan_filled because the "
                "filled grid cannot retain per-device encounter provenance.")

        # if no matching block ids
        if len(block_list) == 0:
            if block_source != "block_info" and \
                    (isinstance(return_nan_filled, np.ndarray) or return_nan_filled):
                period_ns = (10 ** 18) / self._measures[measure_id]['freq_nhz']
                expected_num_values = round((end_time_n - start_time_n) / period_ns)
                return [], np.full(expected_num_values, np.nan, dtype=np.float64)

            return [], np.array([]), np.array([])

        if isinstance(return_nan_filled, np.ndarray) or return_nan_filled:
            return self.get_data_from_blocks(block_list, filename_dict, start_time_n, end_time_n, analog, time_type,
                                             return_nan_gap=return_nan_filled)

        # Read and decode the blocks.
        headers, r_times, r_values = self.get_data_from_blocks(block_list, filename_dict, start_time_n,
                                                               end_time_n, analog, time_type, sort=False,
                                                               allow_duplicates=allow_duplicates)

        if patient_id is not None:
            ranges = self.sql_handler.get_device_time_ranges_by_patient(
                patient_id, end_time_n, start_time_n)
            patient_mask = patient_sample_mask(
                r_times, headers, block_list, clip_patient_ranges(ranges, start_time_n, end_time_n))
            r_times = r_times[patient_mask]
            r_values = r_values[patient_mask]

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            if patient_id is not None:
                order = np.argsort(r_times, kind='stable')
                r_times, r_values = r_times[order], r_values[order]
                if not allow_duplicates:
                    r_times, r_values = collapse_duplicate_times(
                        r_times, r_values, keep=self._duplicate_keep(duplicate_keep))
            else:
                r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n,
                                              allow_duplicates,
                                              duplicate_keep=self._duplicate_keep(duplicate_keep))

        # Convert time data from nanoseconds to unit of choice
        if time_units != 'ns':
            r_times = r_times / time_unit_options[time_units]

        if decode_string_values:
            r_values = string_dict.decode(np.asarray(r_values).astype(np.int64))

        return headers, r_times, r_values

    def get_string_data(self, measure_id: int = None, start_time_n: int = None, end_time_n: int = None,
                        device_id: int = None, patient_id=None, time_type=1, block_info=None,
                        time_units: str = None, sort=True, allow_duplicates=True, measure_tag: str = None,
                        freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                        device_tag: str = None, mrn: str = None, duplicate_keep: str = None):
        """
        .. _get_string_data_label:

        Read string values from a string-typed measure.

        This is the dedicated reader for string measures. ``get_data`` also returns
        decoded strings by default; this two-value return form remains convenient
        when block headers are not needed. Internally it reads the
        int64 dictionary codes via :meth:`get_data` (``analog=False``, no NaN-fill,
        which the numeric core can represent) and decodes them back to strings via
        the measure's :class:`MeasureStringDictionary`. String reads have their own
        getter because :meth:`get_data`'s numeric core -- analog scaling and the
        float NaN-fill buffer -- cannot represent text; the accepted selectors
        mirror :meth:`get_data` exactly.

        :param int measure_id: The measure identifier. If None, measure_tag must be provided.
        :param int start_time_n: Start epoch (inclusive) in units of ``time_units``.
        :param int end_time_n: End epoch (exclusive) in units of ``time_units``.
        :param int device_id: Device identifier (or use device_tag).
        :param int patient_id: Patient identifier (or use mrn).
        :param time_type: Time representation to return, as in :meth:`get_data`.
        :param block_info: Optional pre-fetched block info, as in :meth:`get_data`.
        :param str time_units: Unit for the returned time array. One of ["s","ms","us","ns"].
        :param bool sort: Whether to sort the returned data by time.
        :param bool allow_duplicates: Whether duplicate timestamps may appear in the result.
            Default True (every stored value). False collapses them to exactly one value per
            timestamp, with identical semantics to :meth:`get_data` -- duplicate means same
            timestamp, and the survivor follows the dataset's ``overwrite`` merge conflict policy
            (newest write wins by default, existing value wins under ``'protect'``). Duplicates
            are collapsed on the stored dictionary codes before decoding, so the returned strings
            are always the survivors' own text.
        :param str duplicate_keep: Override the survivor rule for this call: ``"last"`` or
            ``"first"``. See :meth:`get_data`.
        :param str measure_tag: Signal tag; required if measure_id is None.
        :param freq: Sample frequency, helpful with measure_tag.
        :param str units: Measure units, helpful with measure_tag.
        :param str freq_units: Frequency units for ``freq``.
        :param str device_tag: Device tag (exclusive with device_id).
        :param str mrn: Medical record number (exclusive with patient_id).

        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        :returns: ``(times, values)`` where ``times`` is the 1D time array and
            ``values`` is a 1D object ndarray of ``str``.

        Example:

            >>> import numpy as np
            >>> sdk = AtriumSDK(dataset_location="./example_dataset")
            >>> measure_id = sdk.insert_measure(measure_tag="alarm_text", freq=1.0, freq_units="Hz")
            >>> device_id = sdk.insert_device(device_tag="test_device")
            >>> # Strings are written with the ordinary write methods.
            >>> sdk.write_time_value_pairs(
            ...     measure_id, device_id, np.array([0.0, 1.0, 2.0]),
            ...     ["ASYSTOLE", "V-TACH", "ASYSTOLE"], time_units="s")
            >>> times, values = sdk.get_string_data(
            ...     measure_id, start_time_n=0, end_time_n=10, device_id=device_id, time_units="s")
            >>> values
            array(['ASYSTOLE', 'V-TACH', 'ASYSTOLE'], dtype=object)

        .. note::

            String measures are decoded by :meth:`get_data` with its default ``analog=True`` as well.
            ``return_nan_filled`` still raises because no string grid has been defined. :meth:`get_iterator` also supports
            string measures, but a window carries the raw int64 dictionary codes; decode them with
            ``Window.decode_string_signal(...)``.
        """
        # analog=False so the numeric read path returns the raw int64 codes; the
        # guard rails in get_data therefore pass for this call.
        headers, times, codes = self.get_data(
            measure_id=measure_id, start_time_n=start_time_n, end_time_n=end_time_n,
            device_id=device_id, patient_id=patient_id, time_type=time_type, analog=False,
            block_info=block_info, time_units=time_units, sort=sort, allow_duplicates=allow_duplicates,
            measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units,
            device_tag=device_tag, mrn=mrn, return_nan_filled=False, duplicate_keep=duplicate_keep)

        # Resolve the measure_id the same way get_data did, so we load the matching
        # dictionary (get_data accepts either measure_id or measure_tag+freq+units).
        resolved_measure_id = measure_id
        if resolved_measure_id is None:
            resolved_measure_id = get_best_measure_id(self, measure_tag, freq, units, freq_units)
        resolved_measure_id = int(resolved_measure_id)

        if codes.size == 0:
            return times, np.array([], dtype=object)

        string_dict = MeasureStringDictionary.load(self._meta_dir, resolved_measure_id)
        self._check_string_dictionary_not_lost(
            resolved_measure_id, string_dict, self._established_value_type(resolved_measure_id),
            self.sql_handler.get_string_dict_watermark(resolved_measure_id))
        values = string_dict.decode(np.asarray(codes).astype(np.int64))
        return times, values

    def _require_string_measure(self, measure_id: int) -> "MeasureStringDictionary":
        """Validate that ``measure_id`` is a string/event measure and return its
        loaded :class:`MeasureStringDictionary`. Raises a clear ``ValueError`` for a
        missing measure or a numeric (``value_type != 'string'``) measure -- events
        are always string measures.

        The error distinguishes the two ways a caller lands here, because the fix
        is completely different: a measure that genuinely holds numbers (nothing
        to do but pick another measure) versus one that holds text but was never
        classified -- ``value_type`` is nullable, and a measure created by an
        ingest pipeline reads as ``numeric`` until its first string write or an
        explicit :meth:`update_measure`.

        The id is coerced through :func:`require_measure_id` first, so every entry point
        onto this surface (:meth:`get_measure_string_vocabulary`,
        :meth:`get_string_values_present`, :meth:`get_event_intervals`) rejects a measure
        tag with the same message instead of one of them failing differently."""
        measure_id = require_measure_id(measure_id, "measure_id")
        info = self.get_measure_info(measure_id)
        if info is None:
            raise ValueError(f"Measure {measure_id} does not exist.")
        signal_kind, value_type = measure_kind_of(info)
        if not is_string_value_type(value_type):
            info = self.get_measure_info(measure_id) or {}
            described = f"Measure {measure_id}"
            if info.get('tag'):
                described += f" ('{info['tag']}')"
            if self._established_value_type(measure_id) is None:
                # No data committed yet, so nothing is actually established: the
                # 'numeric' above is only the read-time default for a NULL column.
                remedy = (f"This measure has no data yet, so {value_type!r} is only the default "
                          f"for an unset value_type. If it is meant to hold text, classify it "
                          f"first with sdk.update_measure({measure_id}, value_type='string') "
                          f"(and signal_kind='event' or 'state'), or just write string values to "
                          f"it -- the first string write establishes the type.")
            else:
                remedy = ("This measure holds numeric data, which has no string vocabulary to "
                          "pair on. Query the string measure that carries the event text "
                          "instead; sdk.get_all_measures() reports each measure's value_type.")
            raise ValueError(
                f"{described} has value_type {value_type!r}, not 'string'. "
                f"Event / string queries (vocabulary, values-present, event intervals) "
                f"operate on string measures only. {remedy}")
        return MeasureStringDictionary.load(self._meta_dir, measure_id)

    def _resolve_event_source(self, device_id, patient_id, device_tag, mrn):
        """Resolve a (device_id, patient_id) source from any accepted selector.
        Exactly one axis need be given; device_tag/mrn are resolved to ids. Raises
        if nothing identifies a source or a tag/mrn is unknown."""
        if device_id is None and device_tag is not None:
            device_id = self.get_device_id(device_tag)
            if device_id is None:
                raise ValueError(f"device_tag {device_tag!r} not found in the dataset.")
        if patient_id is None and mrn is not None:
            patient_id = self.get_patient_id(mrn)
            if patient_id is None:
                raise ValueError(f"mrn {mrn!r} not found in the dataset.")
        if device_id is None and patient_id is None:
            raise ValueError(
                "A data source is required: pass one of device_id, device_tag, "
                "patient_id, or mrn.")
        return device_id, patient_id

    # The pure interval algebra lives in :mod:`atriumdb.event_intervals` (pairing,
    # container clipping, window union). These stay as methods because they are
    # part of the established internal surface, but they are one-line delegations
    # so the algorithm has exactly one implementation.
    _union_windows = staticmethod(union_windows)

    def _collect_device_patient_windows(self, device_id, patient_id, start_n, end_n):
        """Device<->patient mapping spans for the source, clipped to [start,end].
        Empty list when the ``device_patient`` table has no rows for this source;
        the whole event path must run against an empty ``device_patient`` table."""
        rows = self.get_device_patient_data(
            device_id_list=[device_id] if device_id is not None else None,
            patient_id_list=[patient_id] if patient_id is not None else None,
            start_time=start_n, end_time=end_n, time_units="ns")
        return clip_spans_and_union(((row[2], row[3]) for row in rows), start_n, end_n)

    def _collect_encounter_windows(self, device_id, patient_id, start_n, end_n):
        """Encounter (admission-level) spans for the source, clipped to [start,end].
        For a patient source the patient is used directly; for a device source the
        patient(s) are resolved through ``device_patient`` first (so a device source
        yields no encounter windows when device_patient is empty)."""
        patient_ids = set()
        if patient_id is not None:
            patient_ids.add(patient_id)
        elif device_id is not None:
            for row in self.get_device_patient_data(
                    device_id_list=[device_id], start_time=start_n, end_time=end_n,
                    time_units="ns"):
                patient_ids.add(row[1])
        spans = [(enc[3], enc[4])
                 for pid in patient_ids
                 for enc in self.get_encounters(patient_id=pid, start_time=start_n,
                                                end_time=end_n, time_units="ns")]
        return clip_spans_and_union(spans, start_n, end_n)

    def _resolve_within_windows(self, within, device_id, patient_id, start_n, end_n):
        """Resolve the ``within`` container to a list of disjoint [start,end] ns
        windows plus a label, per the cascade
        ``device_patient -> encounter -> whole-stream``.

        A caller may force a level (``"device_patient" | "encounter" | "none"``);
        ``None`` runs the graceful cascade. When requested/needed scoping data is
        missing we ``warnings.warn`` and fall through rather than silently dropping
        the query -- and the whole path runs with an empty device_patient table."""
        whole = [[start_n, end_n]]

        if within == "none":
            return whole, "whole-stream"

        if within == "device_patient":
            dp = self._collect_device_patient_windows(device_id, patient_id, start_n, end_n)
            if dp:
                return dp, "device_patient"
            warnings.warn(
                "within='device_patient' was requested but no device_patient mapping "
                "is populated for this source; falling back to the whole-stream range.")
            return whole, "whole-stream"

        if within == "encounter":
            enc = self._collect_encounter_windows(device_id, patient_id, start_n, end_n)
            if enc:
                return enc, "encounter"
            warnings.warn(
                "within='encounter' was requested but no encounter data is available "
                "for this source; falling back to the whole-stream range.")
            return whole, "whole-stream"

        if within is None:
            dp = self._collect_device_patient_windows(device_id, patient_id, start_n, end_n)
            if dp:
                return dp, "device_patient"
            enc = self._collect_encounter_windows(device_id, patient_id, start_n, end_n)
            if enc:
                warnings.warn(
                    "device_patient scoping data is empty for this source; falling back "
                    "to encounter scoping in the within cascade.")
                return enc, "encounter"
            warnings.warn(
                "Neither device_patient nor encounter scoping data is available for this "
                "source; falling back to whole-stream scoping (the query range).")
            return whole, "whole-stream"

        raise ValueError(
            f"Unknown within option {within!r}; expected None (cascade), "
            f"'device_patient', 'encounter', or 'none' (whole-stream).")

    # Pure pairing / clipping math -- see :mod:`atriumdb.event_intervals` for the
    # documented implementation and the interval + censoring conventions.
    _pair_from_to = staticmethod(pair_from_to)
    _clip_intervals_to_containers = staticmethod(clip_intervals_to_containers)

    def _collapse_event_intervals(self, times, codes, from_code, to_code,
                                  range_start, range_end, windows):
        """Pair ``from``/``to`` on the full event stream over the query range, then
        clip to the ``within`` container windows. See
        :func:`atriumdb.event_intervals.collapse_event_intervals`."""
        return collapse_event_intervals(times, codes, from_code, to_code,
                                        range_start, range_end, windows)

    def get_measure_string_vocabulary(self, measure_id: int) -> list:
        """Return EVERY string value ever written to a string measure, in code order.

        Reads the per-measure dictionary file via :class:`MeasureStringDictionary`
        with **no data scan** -- cost is bounded by the vocabulary size, not by the
        number of samples stored. Raises a clear ``ValueError`` for a numeric
        measure (events are always string measures).

        :param int measure_id: A string-typed measure.
        :rtype: list[str]
        :returns: All distinct strings ever written, ordered by their int64 code.
        """
        string_dict = self._require_string_measure(measure_id)
        return string_dict.vocabulary()

    def get_string_values_present(self, measure_id: int, start_time, end_time,
                                  device_id: int = None, patient_id=None,
                                  device_tag: str = None, mrn: str = None,
                                  time_units: str = None) -> list:
        """Return the sorted distinct string values actually present for a source over
        the half-open range ``[start_time, end_time)``.

        Unlike :meth:`get_measure_string_vocabulary` (all values ever written), this
        reads the codes for the source over the window (via :meth:`get_string_data`)
        and uniques the values genuinely observed -- "what events occurred for device
        X last week". Raises for a numeric measure.

        :param int measure_id: A string-typed measure.
        :param start_time: Range start (inclusive), in ``time_units``.
        :param end_time: Range end (exclusive), in ``time_units``.
        :param int device_id: Device source (or ``device_tag``).
        :param patient_id: Patient source (or ``mrn``).
        :param str device_tag: Device tag (exclusive with device_id).
        :param str mrn: Medical record number (exclusive with patient_id).
        :param str time_units: Unit of ``start_time``/``end_time`` ("s"/"ms"/"us"/"ns", default "ns").
        :rtype: list[str]
        :returns: Sorted distinct strings observed for the source in the window.
        """
        if start_time is None or end_time is None:
            raise ValueError(
                "start_time and end_time are required for get_string_values_present -- "
                "they bound the window to scan.")
        self._require_string_measure(measure_id)
        device_id, patient_id = self._resolve_event_source(device_id, patient_id, device_tag, mrn)
        _times, values = self.get_string_data(
            measure_id=measure_id, start_time_n=start_time, end_time_n=end_time,
            device_id=device_id, patient_id=patient_id, time_units=time_units)
        if values.size == 0:
            return []
        return sorted({str(v) for v in values.tolist()})

    def get_event_intervals(self, measure: int, from_value: str, to_value: str,
                            device_id: int = None, patient_id=None,
                            device_tag: str = None, mrn: str = None,
                            start_time=None, end_time=None, within=None,
                            time_units: str = None) -> list:
        """Derive ``from_value -> to_value`` state intervals for a string/event measure.

        Both values are strings drawn from the SAME measure's vocabulary: an opening
        event and a closing event are two values of one event stream, not two measures.

        **Collapse pairing.** A run of consecutive ``from`` events, with no intervening
        ``to``, produces ONE interval: it opens at the FIRST ``from`` of the run and
        closes at the first ``to`` that follows. Repeated ``from`` events therefore do
        not restart or nest the interval, and the returned intervals never overlap.
        (Example: ``START, START, START, STOP`` yields a single interval from the first
        ``START`` to the ``STOP``.) Pairing is vectorized -- ``searchsorted`` plus
        ``np.unique`` -- with no per-event Python loop.

        **Containment.** Intervals are scoped to a container, chosen by a cascade:
        ``device_patient`` mappings when that table is populated for the source,
        otherwise ``encounter`` spans, otherwise the whole queried stream. Force a level
        with ``within="device_patient" | "encounter" | "none"``; ``None`` runs the
        cascade. When requested scoping data is missing the method warns and falls
        through to the next level rather than silently returning nothing, and the whole
        path is valid against an entirely empty ``device_patient`` table. A pair whose
        ``from`` and ``to`` land in different containers is SPLIT at the container
        boundary -- it never spans the gap between them -- and each resulting piece is
        flagged censored on the side that was cut.

        **Censoring.** A boundary is never fabricated. A ``from`` with no following
        ``to`` inside its container yields ``end_censored=True`` with ``end_time_n``
        clipped to the container's end; a ``to`` with no preceding ``from`` (the
        container opened while the state was already active) yields
        ``start_censored=True`` with ``start_time_n`` clipped to the container's start.
        The timestamps are always real container boundaries, so a caller that cannot
        tolerate an inferred boundary filters on the two flags.

        This method returns intervals only; it does not rasterize them onto a grid.

        Example usage:

            >>> # A run of START markers closes at the next STOP (collapse pairing).
            >>> sdk.get_event_intervals(
            ...     measure=measure_id, from_value="START", to_value="STOP",
            ...     device_id=device_id, start_time=0, end_time=120, time_units="s")
            [{'start_time_n': 30000000000, 'end_time_n': 60000000000,
              'start_censored': False, 'end_censored': False}]

        :param int measure: The string measure id whose vocabulary ``from_value`` and
            ``to_value`` belong to. An id, not a tag -- resolve a tag first with
            :meth:`get_measure_id`.
        :param str from_value: The opening event string (must be in the vocabulary).
        :param str to_value: The closing event string (must be in the vocabulary).
        :param int device_id: Device source (or ``device_tag``).
        :param patient_id: Patient source (or ``mrn``).
        :param str device_tag: Device tag (exclusive with device_id).
        :param str mrn: Medical record number (exclusive with patient_id).
        :param start_time: Query range start (inclusive), in ``time_units``. Required.
        :param end_time: Query range end (exclusive), in ``time_units``. Required.
        :param within: Container scoping; ``None`` (cascade), ``"device_patient"``,
            ``"encounter"``, or ``"none"`` (whole-stream).
        :param str time_units: Unit of the INPUT ``start_time``/``end_time``
            ("s"/"ms"/"us"/"ns", default "ns"). Returned ``*_time_n`` values are always
            nanoseconds.
        :rtype: list[dict]
        :returns: List of ``{"start_time_n", "end_time_n", "start_censored",
            "end_censored"}`` dicts, sorted by start time, all in nanoseconds.
        """
        time_units = "ns" if time_units is None else time_units
        if time_units not in time_unit_options:
            raise ValueError(f"Invalid time units. Expected one of: {list(time_unit_options.keys())}")
        if start_time is None or end_time is None:
            raise ValueError(
                "start_time and end_time are required for get_event_intervals -- they "
                "bound the whole-stream container and define where censoring clips.")
        if from_value == to_value:
            raise ValueError(
                "from_value and to_value must differ; get_event_intervals pairs an "
                "opening event with a distinct closing event.")
        start_n = int(start_time * time_unit_options[time_units])
        end_n = int(end_time * time_unit_options[time_units])

        measure_id = require_measure_id(measure, "measure")
        string_dict = self._require_string_measure(measure_id)

        from_code = string_dict.code_for(from_value)
        if from_code is None:
            raise ValueError(
                f"from_value {from_value!r} is not in the string vocabulary of measure "
                f"{measure_id}. Known values come from get_measure_string_vocabulary().")
        to_code = string_dict.code_for(to_value)
        if to_code is None:
            raise ValueError(
                f"to_value {to_value!r} is not in the string vocabulary of measure "
                f"{measure_id}. Known values come from get_measure_string_vocabulary().")

        device_id, patient_id = self._resolve_event_source(device_id, patient_id, device_tag, mrn)

        # Read the source's int64 codes over the range (analog=False -> raw codes,
        # the same path get_string_data reads before decoding to strings).
        _headers, times, codes = self.get_data(
            measure_id=measure_id, start_time_n=start_n, end_time_n=end_n,
            device_id=device_id, patient_id=patient_id, analog=False,
            time_units="ns", sort=True)
        times = np.asarray(times, dtype=np.int64).reshape(-1)
        codes = np.asarray(codes).astype(np.int64).reshape(-1)

        windows, _container_label = self._resolve_within_windows(
            within, device_id, patient_id, start_n, end_n)

        intervals = self._collapse_event_intervals(
            times, codes, int(from_code), int(to_code), start_n, end_n, windows)

        return [
            {"start_time_n": s, "end_time_n": e,
             "start_censored": sc, "end_censored": ec}
            for (s, e, sc, ec) in intervals
        ]

    # ------------------------------------------------------------------ #
    # Block-level and remote (API mode) read paths
    # ------------------------------------------------------------------ #
    def get_data_from_blocks(self, block_list, filename_dict, start_time_n, end_time_n, analog=True,
                             time_type=1, sort=True, allow_duplicates=True, return_nan_gap=False,
                             duplicate_keep=None):
        """
        Retrieve data from blocks.

        This method reads data from the specified blocks, decodes it, and returns the headers, times, and values.

        :param list block_list: List of blocks to read data from.
        :param dict filename_dict: Dictionary containing file information.
        :param int start_time_n: Start time of the data to read.
        :param int end_time_n: End time of the data to read.
        :param bool analog: Whether the data is analog or not, defaults to True.
        :param time_type: The type of time returned. Options are:
            - 1: Timestamps (default).
            - 2: Gap array (advanced users only).
            - 'raw': Return as was originally stored.
            - 'encoded': Return in the format currently encoded (usually 2 for periodic signals).
        :param bool sort: Whether to sort the returned data by time. Sorting is only applied when time_type is 1.
        :param bool allow_duplicates: Whether to allow duplicate times in the sorted returned data if they exist. Does
            nothing if sort is false.
        :param bool | ndarray return_nan_gap: Whether or not to return values as a list of nans from start to end.
        :param duplicate_keep: Which copy of a duplicated timestamp survives when
            ``allow_duplicates`` is False -- ``"last"`` (most recently written) or ``"first"``
            (earliest written). ``None`` follows this dataset's merge conflict policy, which is
            what :meth:`get_data` passes.
        :return: Tuple containing headers, times, and values.
        :rtype: tuple
        """
        if self.metadata_connection_type == "api":
            raise ValueError("This function is only meant to work in local mode.")

        # Condense the block list for optimized reading
        read_list = condense_byte_read_list(block_list)

        # Read the data from the files using the read list
        encoded_bytes = self.file_api.read_file_list(read_list, filename_dict)

        # Extract the number of bytes for each block
        num_bytes_list = [row[5] for row in block_list]

        if isinstance(return_nan_gap, np.ndarray) or return_nan_gap:
            return self.block.decode_blocks(
                encoded_bytes, num_bytes_list, analog=True, time_type=1, return_nan_gap=return_nan_gap,
                start_time_n=start_time_n, end_time_n=end_time_n)

        # Decode the data and separate it into headers, times, and values
        r_times, r_values, headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                              time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates,
                                          duplicate_keep=self._duplicate_keep(duplicate_keep))

        return headers, r_times, r_values

    def _duplicate_keep(self, duplicate_keep=None):
        """Which copy of a duplicated timestamp a read keeps, as a ``collapse_duplicate_times``
        ``keep`` value.

        ``None`` derives it from the dataset's merge conflict policy so a read resolves a
        duplicate the same way a write would have: ``'protect'`` keeps the existing (earliest
        written) value, every other policy keeps the newer write's value.
        """
        if duplicate_keep is not None:
            if duplicate_keep not in DUPLICATE_KEEP_OPTIONS:
                raise ValueError(
                    f"duplicate_keep must be one of {DUPLICATE_KEEP_OPTIONS} or None; "
                    f"got {duplicate_keep!r}.")
            return duplicate_keep
        try:
            return "first" if self._merge_conflict_policy() == 'protect' else "last"
        except Exception:
            # A mode without a settings table (e.g. api) still gets the default convention.
            return "last"

    def _get_data_api(self, measure_id: int, start_time_n: int, end_time_n: int, device_id: int = None,
                      patient_id: int = None, mrn: str = None, time_type=1, analog=True, sort=True,
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

        encoded_bytes = self._block_websocket_request(block_info_list)

        # Decode the concatenated bytes to get headers, request times and request values
        r_times, r_values, headers = self.block.decode_blocks(encoded_bytes, num_bytes_list, analog=analog,
                                                              time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort:
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates,
                                          duplicate_keep=self._duplicate_keep())

        return headers, r_times, r_values

    def _block_websocket_request(self, block_info_list):
        if len(block_info_list) == 0:
            return np.array([], dtype=np.uint8)

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

    # ------------------------------------------------------------------ #
    # Writing signal data: the core write path and its helpers
    # ------------------------------------------------------------------ #
    def write_data_easy(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq: int,
                        scale_m: float = None, scale_b: float = None, time_units: str = None, freq_units: str = None,
                        continuous: bool = False):
        """
        .. _write_data_easy_label:

        The simplified method for writing new data to the dataset.

        This method makes it easy to write new data to the dataset by taking care of unit conversions and data type
        handling internally. It supports various time units and frequency units for user convenience.

        .. deprecated::

            ``write_data_easy`` exists only for legacy compatibility and will be removed in a future
            release. It is numeric-only; write new data with :meth:`write_segments` or
            :meth:`write_time_value_pairs` instead.

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
        :param bool continuous: If True, treat this entire call as a single continuous interval in the
            interval index, regardless of internal gaps.
        """

        warnings.warn(
            "write_data_easy is deprecated and retained only for legacy compatibility; it will be removed "
            "in a future release. Use write_segments(...) or write_time_value_pairs(...) instead.",
            DeprecationWarning, stacklevel=2)

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # write_data_easy always chooses a numeric raw_value_type, so string values
        # would reach write_data's string guard and get advice the caller cannot act
        # on ("Omit raw_value_type" -- they never passed one). Reject them here.
        if isinstance(value_data, np.ndarray) and is_string_dtype(value_data.dtype):
            raise ValueError(
                "write_data_easy does not support string values: it is a fixed-frequency "
                "convenience wrapper that always writes a numeric value type. Write string "
                "values with write_time_value_pairs(measure_id, device_id, times, values), "
                "which dictionary-encodes them, and read them back with get_string_data().")

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

        # Determine the raw and encoded value types based on the dtype of value_data
        if np.issubdtype(value_data.dtype, np.integer):
            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        # Call write_data with the determined parameters. encoded_time_type is left
        # unset so write_data auto-selects the time encoding and compression.
        self.write_data(measure_id, device_id, time_data, value_data, freq, int(time_data[0]), raw_time_type=raw_t_t,
                        raw_value_type=raw_v_t, encoded_value_type=encoded_v_t,
                        scale_m=scale_m, scale_b=scale_b, continuous=continuous)

    def write_data(self, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray,
                   freq_nhz: int = None, time_0: int = None, raw_time_type: int = None,
                   raw_value_type: int = None, encoded_time_type: int = None,
                   encoded_value_type: int = None, scale_m: float = None, scale_b: float = None,
                   interval_index_mode: str = None, gap_tolerance: int = None, merge_blocks: bool = True,
                   period_ns: int = None, continuous: bool = False,
                   t_compression: int = None, t_compression_level: int = None):
        """
        .. _write_data_label:

        Advanced method for writing new data to the dataset. This method can be used to express time data as a gap array
        (even sized array, odd values are indices of value_data after a gap and even values are the durations of the
        corresponding gaps in nanoseconds).

        :param int measure_id: Measure identifier corresponding to the measures table in the linked relational database.
        :param int device_id: Device identifier corresponding to the devices table in the linked relational database.
        :param numpy.ndarray time_data: 1D numpy array representing the time information of the data to be written.
        :param numpy.ndarray value_data: 1D numpy array (or list) representing the value information of the data to be
            written. String values are also supported: pass a ``list[str]`` or a string/object numpy array
            (dtype kind ``U``, ``S`` or ``O``) and each value is transparently encoded as an ``int64`` dictionary code
            via the measure's append-only string dictionary (read them back with :meth:`get_string_data`). String
            writes force identity scaling and reject a numeric ``raw_value_type``.
        :param int freq_nhz: Sample frequency, in nanohertz, of the data to be written.
        :param int time_0: Start time of the data to be written.
        :param int raw_time_type: Identifier representing the time format being written, corresponding to the options
            written in the block header.
        :param int raw_value_type: Identifier representing the value format being written, corresponding to the
            options written in the block header. If ``None``, chosen from the value dtype (int64 or double).
        :param int encoded_time_type: Identifier representing how the time information is encoded, corresponding
            to the options written in the block header. If ``None``, auto-chosen from the data (gap array, optionally
            zstd-compressed, or a compressed timestamp array for aperiodic data).
        :param int encoded_value_type: Identifier representing how the value information is encoded, corresponding
            to the options written in the block header. If ``None``, chosen from ``raw_value_type`` (ints are delta
            encoded, floats stored as doubles).
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
            it is treated as a break in continuity or gap in the interval index. If ``None`` (the default), AtriumDB
            chooses a default from the data so that jitter, short dropouts and aperiodic arrival gaps do not flood the
            interval index (see ``choose_interval_gap_tolerance`` and ``widen_gap_tolerance_for_observed_spacing`` for
            the rule). Pass ``0`` to record every gap, or an explicit value to override - an explicit tolerance is
            always honored exactly.
        :param bool merge_blocks: If you're writing data that is less than an optimal block size it will find an already
            existing block that is closest in time to the data you're writing and merge your data with it. Duplicate
            timestamps are resolved by the dataset's ``overwrite`` setting: the new write's values win by default,
            ``'protect'`` keeps the existing values, and ``'error'`` raises instead of merging conflicting data. The
            old block must hold the same kind of data (same raw value type
            and scale factors; if you explicitly requested encoded types, the old block must already use them).
            When the time encoding is being auto-chosen, it is chosen after the merge so it reflects the merged data.
            THIS IS NOT THREAD SAFE and can lead to race conditions if two processes (with two different sdk objects)
            try to ingest (and merge) data for the same measure and device at the same time.
        :param int period_ns: Sampling period in nanoseconds, mutually exclusive with freq_nhz.
        :param bool continuous: If True, record this call's data as a single continuous interval in the interval index,
            regardless of internal gaps. Only the caller's own time span is collapsed; merging with neighboring
            existing data still follows ``gap_tolerance``.
        :param int t_compression: Compression for the time data. If ``None``, it is auto-chosen alongside
            ``encoded_time_type``.
        :param int t_compression_level: Compression level for the time data, used with ``t_compression``.

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
            >>>     measure_id, device_id, gap_arr, value_data, freq_nhz=freq_nhz, time_0=time_zero_nano,
            >>>     raw_time_type=T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
            >>>     raw_value_type=V_TYPE_INT64,
            >>>     encoded_time_type=T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
            >>>     encoded_value_type=V_TYPE_DELTA_INT64)
        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Accept python lists (e.g. list[str]) by normalizing to ndarrays, matching
        # how callers pass values to write_time_value_pairs.
        if not isinstance(value_data, np.ndarray):
            value_data = np.asarray(value_data)

        assert value_data.size > 0, "There are no values in the value array to write. Cannot write no data."
        assert np.issubdtype(time_data.dtype, np.integer), "Time information must be encoded as an integer."

        # Determine the incoming value-kind from the (pre-conversion) dtype, and
        # enforce the numeric/string invariant. A measure is either string-typed
        # or numeric; a conflicting write is rejected instead of being silently
        # accepted and corrupting readability. The first write to an as-yet-empty
        # measure ESTABLISHES its value_type.
        incoming_is_string = is_string_dtype(value_data.dtype)
        established_value_type = None
        if self.dataset_location is not None:
            # Only CHECK for a conflict here (may raise); the value_type is
            # ESTABLISHED after the write commits (see below), so a write that
            # raises downstream cannot poison the measure's value_type.
            established_value_type = self._check_value_type_invariant(int(measure_id), incoming_is_string)

        # String storage: a string/object value array is transparently
        # converted to int64 dictionary codes *before* _resolve_value_types (which
        # would otherwise send it to V_TYPE_DOUBLE). From here on it is an ordinary
        # int64 write -- merge, interval index and time encoding are unchanged
        # because the codes are plain int64 with stable, append-only meaning.
        string_dict = None
        if incoming_is_string:
            if raw_value_type is not None and raw_value_type != V_TYPE_INT64:
                raise ValueError(
                    "String/object value arrays are stored as int64 dictionary codes; a numeric "
                    f"raw_value_type ({VALUE_TYPES_STR.get(raw_value_type, raw_value_type)}) cannot "
                    "be combined with string data. Omit raw_value_type for string writes.")
            # Read the high-water mark BEFORE the dictionary file, not after. Both only
            # ever grow, and the mark is raised only once the entries behind it are
            # durably in the file, so reading in this order guarantees
            # "file length >= mark" whenever nothing has been lost. Reading the mark
            # second would let a concurrent writer's commit land in between and report a
            # perfectly healthy dictionary as truncated.
            dictionary_watermark = self.sql_handler.get_string_dict_watermark(int(measure_id))
            string_dict = MeasureStringDictionary.load(self._meta_dir, int(measure_id))
            # Refuse to hand out codes that history already uses. Must run
            # BEFORE encode(), which is what would silently re-issue code 0.
            self._check_string_dictionary_not_lost(
                int(measure_id), string_dict, established_value_type, dictionary_watermark)
            value_data = string_dict.encode(value_data)
            raw_value_type = V_TYPE_INT64
            # Codes are exact identifiers, never scaled: force identity scaling.
            scale_m = 1.0
            scale_b = 0.0

        # Everything from here on can still fail (encode, file write, SQL). Any
        # failure must also undo the dictionary append made just above, otherwise a
        # rejected batch's free text is retained on disk forever and the measure is
        # permanently established as string-typed by a write that never committed.
        # The append is therefore transactional with the write.
        # ---------------------------------------------------------------- #
        # Serialize the block-merge read-modify-write.
        #
        # A write smaller than one optimal block does not simply insert: it SELECTs the
        # closest existing block, reads and decodes that block's file, merges the new
        # values into it, writes a new file and then deletes the old block row. That is a
        # read-modify-write with no isolation, so two processes writing to the same
        # (measure, device) both merge into the SAME old block and the second one to
        # commit finds the row already deleted -- losing its own values and raising
        # TypeError: 'NoneType' object is not subscriptable out of
        # insert_merged_block_data. Aperiodic/event measures make this the normal case
        # rather than an edge case: every event batch is far below block_size, so events
        # ALWAYS take the merge path. Without this lock, only 65-77 of 100 concurrently
        # written events were readable across four measured trials, with 17-32 raises
        # per trial.
        #
        # The lock is taken only when the merge path is reachable, so ordinary
        # block-sized bulk writes are completely unaffected, and it is keyed per
        # (measure, device), so unrelated streams still write fully in parallel. It is
        # held across the merge, the encode and the commit -- the whole read-modify-write
        # -- and released before the post-commit bookkeeping.
        # ---------------------------------------------------------------- #
        merge_path_reachable = merge_blocks and value_data.size < self.block.block_size
        with ExitStack() as merge_lock:
            if merge_path_reachable:
                merge_lock.enter_context(self._block_merge_lock(measure_id, device_id))
            try:
                freq_nhz, period_ns, _, period_ns_for_calc = self._resolve_write_frequency(
                    freq_nhz, period_ns, raw_time_type, time_data)

                # Explicitly requested encodings are honored exactly; omitted ones are
                # auto-chosen from the data after any block merge.
                requested_encoded_time_type = encoded_time_type
                requested_encoded_value_type = encoded_value_type
                raw_value_type, encoded_value_type = self._resolve_value_types(value_data, raw_value_type, encoded_value_type)

                if encoded_time_type is not None and \
                        not ((raw_time_type == 1 and encoded_time_type == 2) or (raw_time_type == encoded_time_type)):
                    raise ValueError(
                        f"Cannot encode raw time type {TIME_TYPES_STR[raw_time_type]} to encoded time type {TIME_TYPES_STR[encoded_time_type]}")

                # None means "no scaling", stored as m=1.0 / b=0.0 in block headers.
                scale_m = 1.0 if scale_m is None else float(scale_m)
                scale_b = 0.0 if scale_b is None else float(scale_b)

                interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode
                assert interval_index_mode in allowed_interval_index_modes, \
                    f"interval_index must be one of {allowed_interval_index_modes}"

                time_0, measure_id, device_id = int(time_0), int(measure_id), int(device_id)

                time_data, value_data, time_0 = self._sort_write_data(
                    raw_time_type, time_data, value_data, time_0, freq_nhz, period_ns, period_ns_for_calc)

                write_intervals = find_intervals(freq_nhz=freq_nhz, period_ns=period_ns, raw_time_type=raw_time_type,
                                                 time_data=time_data, data_start_time=time_0, num_values=int(value_data.size))
                # A `continuous` write collapses to a single interval bounded by the
                # caller's own data, captured before any block merge so continuity is
                # never asserted over time the caller's data does not cover.
                continuous_bounds = (int(write_intervals[0][0]), int(write_intervals[-1][-1])) if continuous else None

                # Merge writes smaller than one optimal block into the closest existing
                # block, then recompute the intervals so they describe the merged whole.
                old_block = None
                # Deduplication is a side effect of the merge below, so track whether it
                # could have happened at all; when it could not, the write is checked for
                # overlap with existing data and reported instead of silently duplicating
                # it.
                overlap_reason = None
                if merge_blocks and value_data.size < self.block.block_size:
                    old_block, time_data, value_data, time_0, raw_time_type, merge_declined = \
                        self._merge_small_write_with_closest_block(
                            measure_id, device_id, time_data, value_data, time_0, raw_time_type, raw_value_type,
                            freq_nhz=freq_nhz, period_ns=period_ns, scale_m=scale_m, scale_b=scale_b,
                            requested_encoded_time_type=requested_encoded_time_type,
                            requested_encoded_value_type=requested_encoded_value_type)
                    if merge_declined:
                        overlap_reason = ("the existing block it would have merged with holds a "
                                          "different raw value type, scale factors or encoding, or "
                                          "is already full")
                    if old_block is not None:
                        write_intervals = find_intervals(
                            freq_nhz=freq_nhz, period_ns=period_ns, raw_time_type=raw_time_type,
                            time_data=time_data, data_start_time=time_0, num_values=int(value_data.size))
                elif merge_blocks:
                    # A write of a full optimal block or more never takes the merge path,
                    # so nothing deduplicates it no matter what it overlaps.
                    overlap_reason = (
                        f"it holds {int(value_data.size)} values, which is at least one optimal "
                        f"block ({self.block.block_size}), and only writes smaller than that merge "
                        f"with -- and are deduplicated against -- existing blocks")

                if overlap_reason is not None:
                    self._report_undeduplicated_overlap(
                        measure_id, device_id,
                        start_time=time_0,
                        end_time=self._write_end_time(raw_time_type, time_data, value_data.size,
                                                      time_0, freq_nhz, period_ns),
                        num_values=int(value_data.size), reason=overlap_reason)

                if continuous_bounds is not None:
                    write_intervals = collapse_continuous_write_intervals(write_intervals, *continuous_bounds)

                # The observed typical sample spacing of the (possibly merged) data
                # classifies waveform vs aperiodic for the tolerance and encoding choices.
                # It is derived for gap arrays too, so the same aperiodic data gets the
                # same interval gap tolerance whether written as segments or timestamps.
                if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
                    median_delta_ns = observed_median_delta_ns(time_data)
                else:
                    median_delta_ns = observed_median_delta_from_gap_array(
                        time_data, value_data.size, period_ns_for_calc)

                # The density warning only applies when the tolerance was auto-chosen;
                # an explicit tolerance (e.g. 0 to record every gap) is the caller's call.
                gap_tolerance_was_auto = gap_tolerance is None
                gap_tolerance = self._resolve_gap_tolerance(gap_tolerance, period_ns_for_calc, median_delta_ns)

                allow_timestamp = requested_encoded_time_type is None and median_delta_ns is not None \
                    and median_delta_ns > APERIODIC_MIN_PERIOD_NS
                encoded_time_type, t_compression, t_compression_level = self._resolve_time_encoding(
                    encoded_time_type, t_compression, t_compression_level,
                    time_data, value_data, raw_time_type, period_ns_for_calc, allow_timestamp)

                encoded_bytes, encoded_headers, byte_start_array = self.block.encode_blocks(
                    times=time_data, values=value_data, freq_nhz=freq_nhz, period_ns=period_ns, start_ns=time_0,
                    raw_time_type=raw_time_type, raw_value_type=raw_value_type,
                    encoded_time_type=encoded_time_type, encoded_value_type=encoded_value_type,
                    scale_m=scale_m, scale_b=scale_b,
                    t_compression=t_compression, t_compression_level=t_compression_level)

                filename = self.file_api.write_bytes(measure_id, device_id, encoded_bytes)

                block_data, interval_data = get_block_and_interval_data(
                    measure_id, device_id, encoded_headers, byte_start_array, write_intervals,
                    interval_gap_tolerance=gap_tolerance)

                if gap_tolerance_was_auto and len(interval_data) > max(
                        INTERVAL_DENSITY_WARNING_MIN_ROWS, int(INTERVAL_DENSITY_WARNING_RATIO * value_data.size)):
                    _LOGGER.warning(
                        f"Write for measure {measure_id}, device {device_id} produced {len(interval_data)} interval rows "
                        f"for {value_data.size} values. The interval index is meant to give a coarse sense of where data "
                        f"exists; consider a larger gap_tolerance (or continuous=True) if this data is not truly gapped.")

                if old_block is not None:
                    old_tsc_file_name = self.sql_handler.insert_merged_block_data(filename, block_data, old_block,
                                                                                  interval_data, interval_index_mode,
                                                                                  gap_tolerance)
                    # remove the old tsc file from disk if it no longer holds any blocks
                    if old_tsc_file_name is not None:
                        self.file_api.remove(
                            self.file_api.to_abs_path(filename=old_tsc_file_name, measure_id=measure_id, device_id=device_id))
                else:
                    self.sql_handler.insert_tsc_file_data(filename, block_data, interval_data, interval_index_mode,
                                                          gap_tolerance)

                # The blocks are durable, so the dictionary codes they reference must
                # stay: from here on the append is no longer rolled back. Record the
                # vocabulary size those durable blocks may reference, in the metadata
                # database, so a later loss of meta/string_dict/ is detected instead of
                # silently re-issuing the codes they use.
                if string_dict is not None:
                    self._record_string_dictionary_size(int(measure_id), len(string_dict))
                string_dict = None

                # The write has committed: now (and only now) establish/persist the
                # measure's value_type on its first write. Doing this post-commit means a
                # write that raised anywhere above never leaves a poisoning value_type.
                if self.dataset_location is not None:
                    self._establish_value_type(int(measure_id), incoming_is_string)

                return encoded_bytes, encoded_headers, byte_start_array, filename
            except BaseException:
                # Roll back the dictionary append only while the measure holds no committed
                # blocks. With no blocks, no stored data can possibly reference the codes
                # being removed, so reclaiming them is provably safe -- and that is exactly
                # the situation the poisoning bug needs (a measure whose value_type column is
                # still NULL has never completed a write). Once the measure has data, another
                # writer may already have encoded against these codes, and removing them could
                # make committed blocks decode wrongly; the appended strings are left in place
                # instead, where they can no longer establish anything.
                if string_dict is not None and not self.sql_handler.measure_has_blocks(int(measure_id)):
                    try:
                        string_dict.rollback_appends()
                    except Exception as rollback_error:  # pragma: no cover - defensive
                        _LOGGER.error(
                            f"Failed to roll back the string dictionary for measure {measure_id} after a "
                            f"failed write; the dictionary may contain unused codes: {rollback_error}")
                    self._measures.pop(int(measure_id), None)
                raise

    def _resolve_write_frequency(self, freq_nhz, period_ns, raw_time_type, time_data):
        """Resolve the mutually exclusive freq_nhz/period_ns pair for a write,
        detecting the period from timestamp data when neither is given. Returns
        ``(freq_nhz, period_ns, freq_nhz_for_calc, period_ns_for_calc)`` where
        exactly one of the first two is set (preserving which representation the
        caller used) and the *_for_calc values are both always available."""
        if freq_nhz is not None and period_ns is not None:
            raise ValueError("freq_nhz and period_ns are mutually exclusive. Specify only one.")
        if freq_nhz is None and period_ns is None:
            if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
                # detect_period always returns a best-effort value (with warnings if uncertain)
                period_ns = detect_period(time_data)
            else:
                raise ValueError("Either freq_nhz or period_ns must be specified.")

        if period_ns is not None:
            return None, period_ns, int((10 ** 18) // period_ns), period_ns
        return freq_nhz, None, int(freq_nhz), (10 ** 18) // int(freq_nhz)

    @staticmethod
    def _resolve_value_types(value_data, raw_value_type, encoded_value_type):
        """Fill in unspecified value types from the value dtype (ints are delta
        encoded, floats stored as doubles) and validate the combination."""
        if raw_value_type is None:
            raw_value_type = V_TYPE_INT64 if np.issubdtype(value_data.dtype, np.integer) else V_TYPE_DOUBLE
        if encoded_value_type is None:
            encoded_value_type = V_TYPE_DELTA_INT64 if raw_value_type == V_TYPE_INT64 else V_TYPE_DOUBLE
        if not ((raw_value_type == V_TYPE_INT64 and encoded_value_type == V_TYPE_DELTA_INT64)
                or (raw_value_type == encoded_value_type)):
            raise ValueError(
                f"Cannot encode raw value type {VALUE_TYPES_STR[raw_value_type]} to encoded value type "
                f"{VALUE_TYPES_STR[encoded_value_type]}")
        return raw_value_type, encoded_value_type

    @staticmethod
    def _sort_write_data(raw_time_type, time_data, value_data, time_0, freq_nhz, period_ns, period_ns_for_calc):
        """Ensure a write's data is in time order. Timestamp arrays are argsorted
        together with their values; gap arrays with backwards jumps are expanded
        into messages, sorted, and rebuilt (the originals are not modified).
        Returns ``(time_data, value_data, time_0)``."""
        if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            if time_data.size != value_data.size:
                raise ValueError("Time array must be of equal size as the Value array in time type 1.")
            if not np.all(np.diff(time_data) >= 0):
                sorted_indices = np.argsort(time_data)
                time_data = time_data[sorted_indices]
                value_data = value_data[sorted_indices]
        elif raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            # a gap more negative than one period means a message starts before its predecessor
            if not np.all(time_data[1::2] >= -period_ns_for_calc):
                message_starts, message_sizes = reconstruct_messages(
                    time_0, time_data, freq_nhz=freq_nhz, period_ns=period_ns, num_values=int(value_data.size))
                value_data = value_data.copy()
                sort_message_time_values(message_starts, message_sizes, value_data)
                time_0 = int(message_starts[0])
                time_data = create_gap_arr_from_variable_messages(message_starts, message_sizes,
                                                                  freq_nhz=freq_nhz, period_ns=period_ns)
        else:
            raise ValueError("raw_time_type must be either 1 or 2")
        return time_data, value_data, time_0

    @staticmethod
    def _resolve_gap_tolerance(gap_tolerance, period_ns, median_delta_ns):
        """Resolve the interval-index gap tolerance for a write. An explicit value
        always wins; otherwise the default from choose_interval_gap_tolerance is
        widened for aperiodic signals so only gaps far outside the cluster of
        observed arrival spacings split the index (see
        widen_gap_tolerance_for_observed_spacing). Note ``continuous`` does not
        touch the tolerance: it collapses the write's own intervals directly (see
        collapse_continuous_write_intervals), and merging with *existing* data
        still follows this tolerance."""
        if gap_tolerance is None:
            base = choose_interval_gap_tolerance(period_ns)
            return int(widen_gap_tolerance_for_observed_spacing(base, median_delta_ns))
        return int(gap_tolerance)

    def _resolve_time_encoding(self, encoded_time_type, t_compression, t_compression_level,
                               time_data, value_data, raw_time_type, period_ns, allow_timestamp):
        """Fill in an unspecified time encoding and/or time compression from the
        data (see choose_time_encoding): a raw gap array for regular data, a
        zstd gap array for structured deviations, and a zstd timestamp array for
        genuinely aperiodic data. ``allow_timestamp`` gates the last option: it
        is only legal for timestamp-array input and only sensible for signals
        slower than APERIODIC_MIN_PERIOD_NS, and gating it also skips the
        sample-encode size measurement on the fast waveform path."""
        if encoded_time_type is not None and t_compression is not None:
            return encoded_time_type, t_compression, t_compression_level

        if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            # for slow, very irregular data, measure both encodings on a sample
            # with the real codec and keep the smaller one
            measure = (lambda: self._measure_time_encoding_sizes(time_data, period_ns)) if allow_timestamp else None
            chosen_type, chosen_compression, chosen_level = choose_time_encoding(
                period_ns, times_ns=time_data, num_values=int(value_data.size),
                allow_timestamp=allow_timestamp, measure=measure)
        else:
            chosen_type, chosen_compression, chosen_level = choose_time_encoding(
                period_ns, num_gaps=time_data.size // 2, gap_durations=time_data[1::2],
                num_values=int(value_data.size), allow_timestamp=False)

        if encoded_time_type is None:
            encoded_time_type = chosen_type
        if t_compression is None:
            t_compression, t_compression_level = chosen_compression, chosen_level
        return encoded_time_type, t_compression, t_compression_level

    def _merge_small_write_with_closest_block(self, measure_id, device_id, time_data, value_data, time_0,
                                              raw_time_type, raw_value_type, freq_nhz=None, period_ns=None,
                                              scale_m=None, scale_b=None, requested_encoded_time_type=None,
                                              requested_encoded_value_type=None):
        """Try to merge a small write (less than one optimal block worth of values)
        with the closest existing block for this measure/device.

        Returns ``(old_block, time_data, value_data, time_0, raw_time_type, merge_declined)``.
        When a merge happened, ``old_block`` is the block-index row whose data is
        now contained in the returned arrays (the caller must delete it once the
        merged block is written) and the returned arrays/raw time type describe
        the combined data. When no merge happened, ``old_block`` is None and the
        data is returned unchanged.

        ``merge_declined`` distinguishes the two ways ``old_block`` comes back None:
        ``False`` means no candidate block exists near this time range at all (so the
        write cannot be duplicating anything), ``True`` means a candidate WAS found and
        rejected -- wrong raw value type, wrong scale factors, an explicitly requested
        encoding the old block does not use, or a full end block. In that second case
        deduplication silently does not happen, so the caller runs the overlap check
        (see :meth:`_report_undeduplicated_overlap`).

        A merge requires the old block to hold the same kind of data: the same raw
        value type (int/float) and the same scale factors. When the caller
        explicitly requested encoded types, the old block must already use exactly
        those encodings (legacy strict behavior); when the encodings are being
        auto-chosen, they are re-chosen from the merged data after this returns,
        so the old block's encodings don't restrict merging.

        Duplicate timestamps are resolved by the dataset's merge conflict policy
        (see _merge_conflict_policy): the new write's values win by default,
        'protect' keeps the old block's values, and 'error' raises instead of
        merging conflicting data.

        THIS IS NOT THREAD SAFE: two processes merging into the same block at the
        same time can lose one of the writes.
        """
        # Find the end time of the new data so we can find the closest block.
        end_time = self._write_end_time(raw_time_type, time_data, value_data.size, time_0,
                                        freq_nhz, period_ns)

        # find the closest block to the data we are trying to insert
        old_block, end_block = self.sql_handler.select_closest_block(measure_id, device_id, time_0, end_time)
        if old_block is None:
            # Nothing anywhere near this write's time range, so there is nothing to
            # merge with AND nothing it could be duplicating.
            return None, time_data, value_data, time_0, raw_time_type, False

        # If the new data goes on the end and the current end block is already full,
        # start a fresh block instead of re-encoding a full one even bigger.
        if end_block and old_block[8] >= self.block.block_size:
            return None, time_data, value_data, time_0, raw_time_type, True

        # get the file info for the block we are going to merge these values into
        file_info = self.sql_handler.select_file(file_id=old_block[3])
        # Read the encoded data from the files
        encoded_bytes_old = self.file_api.read_file_list([old_block[1:6]],
                                                         filename_dict={file_info[0]: file_info[1]})

        # decode the headers before they are edited by decode blocks so we know the original time type
        header = self.block.decode_headers(encoded_bytes_old, np.array([0], dtype=np.uint64))

        # When the caller explicitly requested encoded types, only merge with a block
        # that already uses them; re-encoding an old block into a different encoding
        # than it was deliberately written with should not happen implicitly.
        if requested_encoded_time_type is not None and header[0].t_encoded_type != requested_encoded_time_type:
            _LOGGER.warning(
                f"The time type ({TIME_TYPES_STR[requested_encoded_time_type]}) you are trying to encode the times as "
                f"doesn't match the encoded time type ({TIME_TYPES_STR[header[0].t_encoded_type]}) of the block "
                f"you are trying to merge with.")
            return None, time_data, value_data, time_0, raw_time_type, True
        if requested_encoded_value_type is not None and header[0].v_encoded_type != requested_encoded_value_type:
            _LOGGER.warning(
                f"The value type ({VALUE_TYPES_STR[requested_encoded_value_type]}) you are trying to encode the values "
                f"as doesn't match the encoded value type ({VALUE_TYPES_STR[header[0].v_encoded_type]}) of the block "
                f"you are trying to merge with.")
            return None, time_data, value_data, time_0, raw_time_type, True

        # The raw value type (int vs float) must match; a block can only hold one.
        if header[0].v_raw_type != raw_value_type:
            _LOGGER.warning(
                f"The raw value type ({VALUE_TYPES_STR[raw_value_type]}) doesn't match the raw value type "
                f"({VALUE_TYPES_STR[header[0].v_raw_type]}) of the block you are trying to merge with.")
            return None, time_data, value_data, time_0, raw_time_type, True

        # make sure the scale factors match. If they don't then don't merge the blocks
        if not (header[0].scale_m == scale_m and header[0].scale_b == scale_b):
            return None, time_data, value_data, time_0, raw_time_type, True

        # if the original time type of the old block is not the same as the time type
        # of the data we are trying to save, we need to make them the same
        if header[0].t_raw_type != raw_time_type:
            if raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
                # the old block is a timestamp array: expand the new gap array to timestamps
                try:
                    time_data = create_timestamps_from_gap_data(values_size=value_data.size, gap_array=time_data,
                                                                start_time=time_0, freq_nhz=freq_nhz,
                                                                period_ns=period_ns)
                    raw_time_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
                except ValueError:
                    freq_desc = f"{freq_nhz}" if freq_nhz is not None else f"period of {period_ns} ns"
                    raise ValueError(f"You are trying to merge a gap array into a block that has the data "
                                     f"saved as a timestamp array and integer timestamps cannot be created "
                                     f"for your gap data with a frequency of {freq_desc}. Either set "
                                     f"merge_blocks to false or pass in the times as a timestamp array.")
            else:
                # the old block is a gap array: convert the new timestamp array to gaps
                time_data = create_gap_arr(time_data, 1, freq_nhz=freq_nhz, period_ns=period_ns)
                raw_time_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Decode the data and get the values and the times we are going to merge this data with
        r_time, r_value, _ = self.block.decode_blocks(encoded_bytes_old, num_bytes_list=[old_block[5]],
                                                      analog=False, time_type=header[0].t_raw_type)

        # cast the new values to the dtype the old block decodes to (int64/float64)
        # so the dtype check in merge_gap_data doesn't fail
        if raw_value_type == V_TYPE_INT64 and value_data.dtype != np.int64:
            value_data = value_data.astype(np.int64)
        elif raw_value_type == V_TYPE_DOUBLE and value_data.dtype != np.float64:
            value_data = value_data.astype(np.float64)

        policy = self._merge_conflict_policy()
        if policy == 'error':
            self._raise_if_merge_conflicts(measure_id, device_id, r_time, r_value.size, int(header[0].start_n),
                                           time_data, value_data.size, time_0, raw_time_type,
                                           freq_nhz=freq_nhz, period_ns=period_ns)

        # Merge the blocks. Both merge functions resolve duplicate timestamps in
        # favor of the data set passed second, so the conflict policy is applied
        # by argument order: new write second (wins) by default, old block second
        # under 'protect'.
        if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            if policy == 'protect':
                time_data, value_data = merge_timestamp_data(value_data, time_data, r_value, r_time)
            else:
                time_data, value_data = merge_timestamp_data(r_value, r_time, value_data, time_data)
            time_0 = int(time_data[0])
        else:
            if policy == 'protect':
                value_data, time_data, time_0 = merge_gap_data(value_data, time_data, time_0,
                                                               r_value, r_time, header[0].start_n,
                                                               freq_nhz=freq_nhz, period_ns=period_ns)
            else:
                value_data, time_data, time_0 = merge_gap_data(r_value, r_time, header[0].start_n,
                                                               value_data, time_data, time_0,
                                                               freq_nhz=freq_nhz, period_ns=period_ns)

        return old_block, time_data, value_data, time_0, raw_time_type, False

    @staticmethod
    def _write_end_time(raw_time_type, time_data, num_values, time_0, freq_nhz=None, period_ns=None):
        """Timestamp of a write's LAST sample, in the same inclusive convention the
        block index stores (``block_index.end_time_n``). Shared by the block-merge
        candidate search and the overlap check so the two cannot disagree about
        where a write ends -- a half-period difference there turns a contiguous
        append into a false overlap report."""
        if raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            # _calc_end_time_from_gap_data returns end_time + one period.
            end_time = _calc_end_time_from_gap_data(values_size=num_values, gap_array=time_data,
                                                    start_time=time_0, freq_nhz=freq_nhz, period_ns=period_ns)
            end_time -= freq_nhz_to_period_ns(freq_nhz) if freq_nhz is not None else period_ns
            return int(end_time)
        return int(time_data[-1])

    def _report_undeduplicated_overlap(self, measure_id, device_id, start_time, end_time,
                                       num_values, reason):
        """Enforce ``overwrite='error'`` for a write that overlaps existing data on a path
        where no deduplication can happen.

        Replay idempotency in this SDK is a *side effect* of the small-write block merge:
        ``write_data`` merges into the closest existing block only when the write is
        smaller than one optimal block, and that merge is what collapses duplicate
        timestamps. A write of ``block_size`` values or more skips the merge entirely and
        is simply appended, so the SAME timestamps can be stored twice.

        That is accepted, deliberately. Write speed is the priority and duplicates are
        expected at live ingest; the write path is not going to decode, merge, re-encode
        and delete N overlapping blocks to guarantee dedup. **Surviving duplicates are a
        READ-side concern**: ``get_data(..., allow_duplicates=False)`` (and
        ``get_string_data``) collapse them to one sample per timestamp, cheaply and
        vectorized. See :meth:`get_data` for the exact semantics.

        So this method is quiet by design. It exists for the one case that still has to be
        enforced at write time -- a dataset configured with ``overwrite='error'``, which
        asks for such a write to be refused rather than stored. Under every other policy
        the overlap is reported at DEBUG level only, and the ``select_blocks`` query is
        skipped entirely unless someone has DEBUG logging on, so ordinary large writes pay
        nothing for it and emit nothing.

        Block-index bounds are the real bounds of stored data (unlike interval rows, which
        are padded by ``gap_tolerance``), so a contiguous append -- the normal bulk-ingest
        case -- finds nothing. Overlap is reported at time-range granularity: two writes
        can share a span without sharing a single timestamp, so the message says
        "overlaps", not "duplicates", and names the span.
        """
        enforcing = self._merge_conflict_policy() == 'error'
        if not enforcing and not _LOGGER.isEnabledFor(logging.DEBUG):
            # Nothing to raise and nothing anyone would see: skip the index query so the
            # write path pays no cost for a condition that is handled on read.
            return

        overlapping = self.sql_handler.select_blocks(
            int(measure_id), int(start_time), int(end_time), device_id=int(device_id))
        if not overlapping:
            return

        existing_values = sum(int(block[8]) for block in overlapping)
        overlap_start = max(int(start_time), min(int(block[6]) for block in overlapping))
        overlap_end = min(int(end_time), max(int(block[7]) for block in overlapping))
        detail = (
            f"Write for measure {measure_id}, device {device_id} covering "
            f"[{int(start_time)}, {int(end_time)}] ns ({num_values} values) overlaps data already "
            f"stored for that stream over [{overlap_start}, {overlap_end}] ns "
            f"({existing_values} values in {len(overlapping)} existing block(s)), and this write "
            f"cannot be deduplicated against it because {reason}.")

        if enforcing:
            raise ValueError(
                detail + " This dataset's overwrite setting is 'error'. Write non-overlapping "
                         "data, or set the overwrite setting to 'overwrite'/'protect' to accept "
                         "the duplication and resolve it on read with "
                         "get_data(..., allow_duplicates=False).")

        _LOGGER.debug(
            detail + " Both copies will be stored; this is expected and not an error. Read with "
                     "get_data(..., allow_duplicates=False) / "
                     "get_string_data(..., allow_duplicates=False) to collapse them to one "
                     "sample per timestamp.")

    @staticmethod
    def _raise_if_merge_conflicts(measure_id, device_id, old_times, old_size, old_start, new_times, new_size,
                                  new_start, raw_time_type, freq_nhz=None, period_ns=None):
        """Enforce the 'error' merge conflict policy: raise if the new write
        shares any timestamp with the block it is about to merge into. For gap
        arrays the exact check needs integer timestamps; when those cannot be
        constructed (imperfect frequency), overlapping time spans are treated as
        a conflict conservatively."""
        if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            conflict = np.intersect1d(old_times, new_times).size > 0
        else:
            old_end = _calc_end_time_from_gap_data(old_size, old_times, old_start,
                                                   freq_nhz=freq_nhz, period_ns=period_ns)
            new_end = _calc_end_time_from_gap_data(new_size, new_times, new_start,
                                                   freq_nhz=freq_nhz, period_ns=period_ns)
            if new_start >= old_end or old_start >= new_end:
                conflict = False
            else:
                try:
                    old_ts = create_timestamps_from_gap_data(old_size, old_times, old_start,
                                                             freq_nhz=freq_nhz, period_ns=period_ns)
                    new_ts = create_timestamps_from_gap_data(new_size, new_times, new_start,
                                                             freq_nhz=freq_nhz, period_ns=period_ns)
                    conflict = np.intersect1d(old_ts, new_ts).size > 0
                except ValueError:
                    conflict = True
        if conflict:
            raise ValueError(
                f"Data to be written for measure {measure_id}, device {device_id} shares timestamps with already "
                f"ingested data and this dataset's overwrite setting is 'error'. Set the dataset's overwrite "
                f"setting to 'overwrite' (new values win) or 'protect' (existing values win), or write "
                f"non-overlapping data.")

    # ------------------------------------------------------------------ #
    # Writing signal data: buffered, segment and time-value-pair entry points
    # ------------------------------------------------------------------ #
    def write_buffer(self, max_values_per_measure_device=None, max_total_values_buffered=None,
                     continuous=False, merge_blocks=True):
        """
        Create a buffer Context Object to batch incoming segments/signals until they hit some threshold,
        are manually flushed to the dataset, or are automatically flushed by exiting the context opened by this object.

        :param int max_values_per_measure_device: (Optional) If the buffer for a measure-device pair ever goes over this number of values,
            the data will be automatically flushed to the dataset. Defaults to 100 blocks.
        :param int max_total_values_buffered: (Optional) If the total number of buffered values across all measure-device pairs
            exceeds this number, the oldest buffer that has values in it will be automatically flushed. Defaults to 10,000 blocks.
        :param bool continuous: (Optional) If True, every flushed batch is treated as a single continuous interval in the
            interval index, regardless of internal gaps.
        :param bool merge_blocks: (Optional, default True) If a flush is smaller than one optimal block, merge it with
            the closest existing block. Pass False to always create new blocks (see write_data).

        Example:

            >>> # Using write_buffer for batched writes
            >>> with sdk.write_buffer(max_values_per_measure_device=100, max_total_values_buffered=1000) as buffer:
            ...     # Write multiple small segments to buffer
            ...     for i in range(5):
            ...         message_values = np.arange(i * 10, (i + 1) * 10)
            ...         start_time = i * 10.0
            ...         sdk.write_segment(measure_id, device_id, message_values, start_time, freq=1.0, freq_units="Hz")
            ...     # Buffer auto-flushes when context is exited

        **Notes:**

        - The buffer will manage sub-buffers for each measure-device combination used within its context.
        - Interval-index settings such as ``gap_tolerance`` belong to ``write_segments`` or
          ``write_time_value_pairs``. A sub-buffer rejects conflicting settings rather than
          silently selecting one.

        """
        return WriteBuffer(
            self,
            max_values_per_measure_device=max_values_per_measure_device,
            max_total_values_buffered=max_total_values_buffered,
            continuous=continuous,
            merge_blocks=merge_blocks,
        )

    def write_segment(self, measure_id: int, device_id: int, segment_values: np.ndarray, start_time: float | int,
                      period: float = None, freq: float = None, time_units: str = None,
                      freq_units: str = None, scale_m: float = None, scale_b: float = None,
                      continuous: bool = False, merge_blocks: bool = True, gap_tolerance: float = None):
        """
        Write a single segment consisting of contiguous values starting at a specific time.

        :param int measure_id: Identifier for the measure, corresponding to the measures table in the linked relational database.
        :param int device_id: Identifier for the device, corresponding to the devices table in the linked relational database.
        :param np.ndarray segment_values: List or 1D numpy array of contiguous values to write.
        :param float start_time: Epoch time when the segment starts. If `time_units` is specified, `start_time` is assumed to be in those units.
        :param float period: (Optional) Sampling period of the data to be written. Only one of `period` or `freq` should be specified.
                             If units other than the default (seconds) are used, specify the desired unit using the `time_units` parameter.
        :param float freq: (Optional) Sampling frequency of the data to be written. Only one of `period` or `freq` should be specified.
                           If units other than the default (hertz) are used, specify the desired unit using the `freq_units` parameter.
        :param str time_units: (Optional) Unit for `start_time` and `period`, which can be one of ["s", "ms", "us", "ns"]. Default is nanoseconds.
        :param str freq_units: (Optional) Unit for `freq`, which can be one of ["Hz", "kHz", "MHz", "GHz"]. Default is hertz.
        :param float scale_m: (Optional) Scaling factor applied to the values (slope in y = mx + b).
        :param float scale_b: (Optional) Offset applied to the values (intercept in y = mx + b).
        :param bool continuous: (Optional) If True, treat this entire call as a single continuous interval in the
            interval index, regardless of internal gaps.
        :param bool merge_blocks: (Optional, default True) If the write is smaller than one optimal block, merge it
            with the closest existing block. Pass False to always create new blocks (see write_data).
        :param float gap_tolerance: (Optional) Interval-index gap policy in ``time_units``. ``None`` chooses
            the data-driven default; ``0`` records every gap.

        Example:

            >>> import numpy as np
            >>> sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
            >>> measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
            >>> device_id = sdk.insert_device(device_tag="test_device")

            >>> # Inserting a single segment
            >>> segment_values = np.arange(50)  # Continuous values from 0 to 49
            >>> start_time = 0.0  # Start time in seconds
            >>> sdk.write_segment(measure_id, device_id, segment_values, start_time, freq=1.0, freq_units="Hz")

        **Notes:**

        - This method is ideal for writing continuous sequences of data that start at a specific time and have uniform sampling intervals.
        - Output from medical monitors, or wfdb Records from physionet dataset typically have this format.
        - If you have multiple segments to write, consider using `write_segments` for better performance.

        """
        # Wrap the single segment and start time into lists to use with write_segments
        segments = [segment_values]
        start_times = [start_time]

        # Call write_segments with the single segment
        self.write_segments(
            measure_id=measure_id,
            device_id=device_id,
            segments=segments,
            start_times=start_times,
            period=period,
            freq=freq,
            time_units=time_units,
            freq_units=freq_units,
            scale_m=scale_m,
            scale_b=scale_b,
            continuous=continuous,
            merge_blocks=merge_blocks,
            gap_tolerance=gap_tolerance,
        )

    def write_segments(self, measure_id: int, device_id: int, segments: List[np.ndarray],
                       start_times: List[float | int],
                       period: float = None, freq: float = None, time_units: str = None,
                       freq_units: str = None, scale_m: float = None, scale_b: float = None,
                       continuous: bool = False, merge_blocks: bool = True, gap_tolerance: float = None):
        """
        Write multiple segments consisting of value arrays and corresponding start times.

        :param int measure_id: Identifier for the measure, corresponding to the measures table in the linked relational database.
        :param int device_id: Identifier for the device, corresponding to the devices table in the linked relational database.
        :param List[ndarray] segments: Each list item is a numpy array of contiguous values that corresponds to a `start_time`
            from an equally sized start_times list.
        :param List[int|float] start_times: Each list item is a float or int representing a start time that corresponds to a `segment`
            from an equally sized segments list.
        :param float period: (Optional) Sampling period of the data to be written. Only one of `period` or `freq` should be specified.
            If units other than the default (seconds) are used, specify the desired unit using the `time_units` parameter.
        :param float freq: (Optional) Sampling frequency of the data to be written. Only one of `period` or `freq` should be specified.
            If units other than the default (hertz) are used, specify the desired unit using the `freq_units` parameter.
        :param str time_units: (Optional) Unit for `start_time` and `period`, which can be one of ["s", "ms", "us", "ns"]. Default is nanoseconds.
        :param str freq_units: (Optional) Unit for `freq`, which can be one of ["Hz", "kHz", "MHz", "GHz"]. Default is hertz.
        :param float scale_m: (Optional) Scaling factor applied to the values (slope in y = mx + b).
            It may be a single number or a list with one number per segment
        :param float scale_b: (Optional) Offset applied to the values (intercept in y = mx + b).
            It may be a single number or a list with one number per segment
        :param bool continuous: (Optional) If True, treat all data written in this call as a single continuous
            interval in the interval index, regardless of gaps between segments.
        :param bool merge_blocks: (Optional, default True) If the write is smaller than one optimal block, merge it
            with the closest existing block. Pass False to always create new blocks (see write_data).
        :param float gap_tolerance: (Optional) Interval-index gap policy in ``time_units``. ``None`` chooses
            the data-driven default; ``0`` records every gap.

        Example:

            >>> import numpy as np
            >>> sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
            >>> measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
            >>> device_id = sdk.insert_device(device_tag="test_device")

            >>> # Inserting multiple segments at once
            >>> segments = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
            >>> start_times = [0.0, 10.0, 20.0]  # Start times in seconds for each segment
            >>> sdk.write_segments(measure_id, device_id, segments, start_times, freq=1.0, freq_units="Hz")

        **Notes:**

        - This method is optimized for batch writing of segments and is more efficient than calling `write_segment` multiple times.

        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Set default time and frequency units if not provided
        time_units = "ns" if time_units is None else time_units
        freq_units = "Hz" if freq_units is None else freq_units
        interval_gap_tolerance_nano = None if gap_tolerance is None else int(
            gap_tolerance * time_unit_options[time_units])

        # Set default for scale factors
        scale_m = 1 if scale_m is None else scale_m
        scale_b = 0 if scale_b is None else scale_b

        # Confirm measure and device information
        measure_info, device_info = self._require_measure_and_device(measure_id, device_id)

        # Figure out the frequency/period - handle mutually exclusive period/freq
        if period is not None and freq is not None:
            raise ValueError("period and freq are mutually exclusive. Specify only one.")

        if freq is not None:
            freq_nano = convert_to_nanohz(freq, freq_units)
            period_ns = None
        elif period is not None:
            period_ns = int(period * time_unit_options[time_units])
            freq_nano = None
        else:
            freq_nano = measure_info["freq_nhz"]
            period_ns = None

        # Create message list for writing.
        scale_m_list = scale_m if isinstance(scale_m, list) else [scale_m] * len(segments)
        scale_b_list = scale_b if isinstance(scale_b, list) else [scale_b] * len(segments)
        write_segments = []
        for values, start_time, m, b in zip(segments, start_times, scale_m_list, scale_b_list):
            if not isinstance(values, np.ndarray):
                raise ValueError(f"Individual segments must be numpy arrays, not {type(values)}")

            if isinstance(start_time, np.generic):
                start_time = start_time.item()

            if not isinstance(start_time, (int, float)):
                raise ValueError(f"Individual start times must be int or float, not {type(start_time)}")
            message_dict = {
                'start_time_nano': int(start_time * time_unit_options[time_units]),
                'values': values,
                'scale_m': m,
                'scale_b': b,
                'freq_nhz': freq_nano,
                'period_ns': period_ns,
                'gap_tolerance_is_explicit': gap_tolerance is not None,
                'interval_gap_tolerance_nano': interval_gap_tolerance_nano,
            }

            write_segments.append(message_dict)

        if self._active_buffer is None:
            # Write immediately to disk
            self._write_segments_to_dataset(measure_id, device_id, write_segments,
                                            interval_gap_tolerance_nano=interval_gap_tolerance_nano, continuous=continuous,
                                            merge_blocks=merge_blocks)
        else:
            # Push new segments to the buffer
            self._active_buffer.push_segments(measure_id, device_id, write_segments, continuous=continuous,
                                              merge_blocks=merge_blocks)

    def _measure_time_encoding_sizes(self, times_ns, period_ns, sample_size=ENCODE_SAMPLE_SIZE):
        """Encode a leading sample of a timestamp array both as a zstd gap array
        and as a zstd timestamp array, returning ``{time_type: compressed_bytes}``.
        This lets the time-encoding choice for very irregular data be made from
        measured sizes rather than a heuristic."""
        sample = times_ns[:sample_size]
        if sample.size < 2:
            return None
        dummy_values = np.zeros(sample.size, dtype=np.int64)
        sizes = {}
        for ett in (T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO):
            try:
                _, headers, _ = self.block.encode_blocks(
                    times=sample, values=dummy_values, period_ns=int(period_ns), start_ns=int(sample[0]),
                    raw_time_type=1, raw_value_type=V_TYPE_INT64, encoded_time_type=ett,
                    encoded_value_type=V_TYPE_DELTA_INT64, scale_m=1.0, scale_b=0.0,
                    t_compression=COMPRESSION_TYPES['ZSTD'], t_compression_level=DEFAULT_TIME_COMPRESSION_LEVEL)
                sizes[ett] = int(sum(h.t_num_bytes for h in headers))
            except Exception:
                sizes[ett] = None
        return sizes

    def _write_segments_to_dataset(self, measure_id, device_id, write_segments, interval_gap_tolerance_nano=None,
                                   continuous=False, merge_blocks=True):
        sorted_segments = sorted(write_segments, key=lambda x: x['start_time_nano'])
        message_start_epoch_array = []
        message_size_array = []

        # Get parameters from first segment
        freq_nhz = sorted_segments[0]['freq_nhz']
        period_ns = sorted_segments[0]['period_ns']
        scale_m = sorted_segments[0]['scale_m']
        scale_b = sorted_segments[0]['scale_b']
        message_dtype = sorted_segments[0]['values'].dtype

        for message in sorted_segments:
            message_start_epoch_array.append(message['start_time_nano'])
            message_size_array.append(message['values'].size)

            # Check consistency
            if message['freq_nhz'] != freq_nhz or message['period_ns'] != period_ns:
                raise ValueError("Segments inserted do not all have the same frequency/period.")

            if message['scale_m'] != scale_m or message['scale_b'] != scale_b:
                raise ValueError("Segments inserted do not all have the same scale factors.")
            if message['values'].dtype != message_dtype:
                raise ValueError("Segments inserted do not all have the same dtype.")

        # Convert segments to gap_data
        gap_data = create_gap_arr_from_variable_messages(
            message_start_epoch_array, message_size_array, freq_nhz=freq_nhz, period_ns=period_ns)

        value_data = np.concatenate([message['values'] for message in sorted_segments])
        time_0 = int(sorted_segments[0]['start_time_nano'])

        self.write_data(measure_id, device_id, gap_data, value_data, freq_nhz=freq_nhz, period_ns=period_ns,
                        time_0=time_0, raw_time_type=T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
                        scale_m=scale_m, scale_b=scale_b, interval_index_mode="merge",
                        gap_tolerance=interval_gap_tolerance_nano, merge_blocks=merge_blocks, continuous=continuous)

    def write_time_value_pairs(self, measure_id: int, device_id: int, times: np.ndarray, values: np.ndarray,
                               period: float = None, freq: float = None, time_units: str = None, freq_units: str = None,
                               scale_m: float = None, scale_b: float = None, continuous: bool = False,
                               merge_blocks: bool = True, gap_tolerance: float = None):
        """
        Write time-value pairs where each value corresponds to a specific timestamp.

        :param int measure_id: Identifier for the measure, corresponding to the measures table in the linked relational database.
        :param int device_id: Identifier for the device, corresponding to the devices table in the linked relational database.
        :param ndarray values: Numpy array (or list) of values to write. String values are supported: pass a
            ``list[str]`` or a string/object numpy array and each value is dictionary-encoded as an ``int64`` code
            (read them back with :meth:`get_string_data`). See the ``value_data`` note on :meth:`write_data`.
        :param ndarray times: Numpy array of corresponding timestamps for each value. The shape of `values` and `times` must match.
        :param float period: (Optional) Sampling period of the data. Only one of `period` or `freq` should be specified.
                             If specified, time deltas in `times` will be adjusted to match `period` within the `gap_tolerance`.
        :param float freq: (Optional) Sampling frequency of the data. Only one of `period` or `freq` should be specified.
                           If specified, time deltas in `times` will be adjusted based on `freq` within the `gap_tolerance`.
        :param str time_units: (Optional) Unit for `times` and `period`, which can be one of ["s", "ms", "us", "ns"]. Default is nanoseconds.
        :param str freq_units: (Optional) Unit for `freq`, which can be one of ["Hz", "kHz", "MHz", "GHz"]. Default is hertz.
        :param float scale_m: (Optional) Scaling factor applied to the values (slope in y = mx + b). Default is 1.0.
        :param float scale_b: (Optional) Offset applied to the values (intercept in y = mx + b). Default is 0.0.
        :param bool continuous: (Optional) If True, treat this entire call as a single continuous interval in the
            interval index, regardless of internal gaps.
        :param bool merge_blocks: (Optional, default True) If the write is smaller than one optimal block, merge it
            with the closest existing block. Pass False to always create new blocks (see write_data).
        :param float gap_tolerance: (Optional) Interval-index gap policy in ``time_units``. ``None`` chooses
            the data-driven default; ``0`` records every gap.

        Example:

            >>> import numpy as np
            >>> sdk = AtriumSDK.create_dataset(dataset_location, db_type, connection_params)
            >>> measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1.0, freq_units="Hz")
            >>> device_id = sdk.insert_device(device_tag="test_device")

            >>> # Inserting time-value pairs
            >>> times = np.array([0.0, 2.0, 4.5])  # Time values in seconds
            >>> values = np.array([100, 200, 300])  # Corresponding values
            >>> sdk.write_time_value_pairs(measure_id, device_id, times, values)

            >>> # String values work the same way (dictionary-encoded automatically)
            >>> str_times = np.array([0.0, 2.0, 4.5])
            >>> str_values = ["ASYSTOLE", "V-TACH", "ASYSTOLE"]
            >>> sdk.write_time_value_pairs(measure_id, device_id, str_times, str_values, time_units="s")
            >>> read_times, read_values = sdk.get_string_data(
            ...     measure_id, start_time_n=0, end_time_n=10, device_id=device_id, time_units="s")

        **Notes:**

        - If neither `freq` nor `period` is specified, the method will attempt to infer the sampling frequency from the most common difference between consecutive timestamps in `times`.
        - A write of a SINGLE time-value pair has no consecutive timestamps to infer from, so the measure's own declared
          period (``get_measure_info(measure_id)['period_ns']``) is used instead. Pass `freq` or `period` explicitly to
          store something else.
        - Use this method when dealing with irregularly sampled data or if your data is already formatted in time-value pairs.

        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for writing data.")

        # Normalize list inputs to ndarrays so callers can pass e.g. a list[str] of
        # values or a list of timestamps, matching write_data. String/object value
        # arrays flow through unchanged and are dictionary-encoded in write_data.
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
        if not isinstance(times, np.ndarray):
            times = np.asarray(times)

        # Collapse fixed-width unicode/bytes value arrays to dtype=object. numpy
        # sizes '<U2' for ["OK"] and '<U8' for ["ASYSTOLE"], so two ordinary alarm
        # strings would otherwise look like two different data types to the buffered
        # flush's dtype-consistency check and abort the whole flush.
        # Every real event stream has variable-length text; object is the dtype the
        # string write path uses anyway.
        if is_fixed_width_string_dtype(values.dtype):
            values = values.astype(object)

        if values.size == 0:
            return

        if values.shape != times.shape:
            raise ValueError("values and times must be numpy arrays of equal shape.")

        # Set default time and frequency units if not provided
        time_units = "ns" if time_units is None else time_units
        freq_units = "Hz" if freq_units is None else freq_units
        interval_gap_tolerance_nano = None if gap_tolerance is None else int(
            gap_tolerance * time_unit_options[time_units])

        # Set default for scale factors
        scale_m = 1 if scale_m is None else scale_m
        scale_b = 0 if scale_b is None else scale_b

        # Confirm measure and device information
        measure_info, device_info = self._require_measure_and_device(measure_id, device_id)

        # Convert times to nanoseconds
        if time_units != "ns":
            times_ns_array = convert_to_nanoseconds(times, time_units)
        else:
            times_ns_array = times.astype(np.int64)

        # Figure out the frequency/period
        if period is not None and freq is not None:
            raise ValueError("period and freq are mutually exclusive. Specify only one.")

        # If neither is provided, attempt to detect (or defer to flush time if buffering)
        if period is None and freq is None:
            if self._active_buffer is not None:
                # When buffering, allow null period - detection will happen at flush time
                # when we have more data to work with.
                freq_nano = None
                period_ns = None
            else:
                nominal_period_ns = get_measure_period_ns(measure_info) \
                    if len(times) < 2 else None
                if nominal_period_ns is not None:
                    period_ns = nominal_period_ns
                    freq_nano = None
                else:
                    detected_period = detect_period(times)
                    period = detected_period
                    period_ns = convert_to_nanoseconds(period, time_units)
                    freq_nano = None
        elif freq is not None:
            freq_nano = convert_to_nanohz(freq, freq_units)
            period_ns = None
        else:
            # period is not None
            period_ns = convert_to_nanoseconds(period, time_units)
            freq_nano = None

        # Create data dictionary
        data_dict = {
            'times': times_ns_array,
            'values': values,
            'scale_m': scale_m,
            'scale_b': scale_b,
            'freq_nhz': freq_nano,
            'period_ns': period_ns,
            'gap_tolerance_is_explicit': gap_tolerance is not None,
            'interval_gap_tolerance_nano': interval_gap_tolerance_nano,
        }

        if self._active_buffer is None:
            # Ingest Immediately
            self._write_time_value_pairs_to_dataset(measure_id, device_id, [data_dict],
                                                    interval_gap_tolerance_nano=interval_gap_tolerance_nano, continuous=continuous,
                                                    merge_blocks=merge_blocks)
        else:
            # Push data to buffer
            self._active_buffer.push_time_value_pairs(measure_id, device_id, data_dict, continuous=continuous,
                                                      merge_blocks=merge_blocks)

    def _write_time_value_pairs_to_dataset(self, measure_id, device_id, data_dicts, interval_gap_tolerance_nano=None,
                                           continuous=False, merge_blocks=True):
        # Get parameters from first data dict
        freq_nhz = data_dicts[0]['freq_nhz']
        period_ns = data_dicts[0]['period_ns']
        scale_m = data_dicts[0]['scale_m']
        scale_b = data_dicts[0]['scale_b']
        data_dtype = data_dicts[0]['values'].dtype

        # Ensure consistency across data_dicts (allow None == None for deferred detection)
        for data in data_dicts:
            if data['freq_nhz'] != freq_nhz or data['period_ns'] != period_ns:
                # Both being None is consistent (deferred detection case from buffering)
                if not (data['freq_nhz'] is None and freq_nhz is None
                        and data['period_ns'] is None and period_ns is None):
                    raise ValueError("Data dictionaries have inconsistent frequencies/periods.")
            if data['scale_m'] != scale_m or data['scale_b'] != scale_b:
                raise ValueError("Data dictionaries have inconsistent scale factors.")
            # Compare the dtype KIND, not the exact dtype: all string-like arrays
            # ('U'/'S'/'O') are one data type as far as the write path is concerned
            # (they all become dictionary codes), and numpy's per-array width for
            # unicode arrays is an artifact of the batch, not of the data.
            if not same_write_value_kind(data['values'].dtype, data_dtype):
                raise ValueError(
                    f"Data dictionaries have inconsistent data types "
                    f"({data['values'].dtype} vs {data_dtype}); string and numeric values "
                    f"cannot be written together.")

        # Combine times and values. np.unique keeps the first occurrence of each
        # duplicate timestamp, so push order decides who wins, following the
        # dataset's merge conflict policy: newest push first (newest wins) by
        # default, oldest push first under 'protect'.
        ordered_dicts = data_dicts if self._merge_conflict_policy() == 'protect' else list(reversed(data_dicts))
        all_times = np.concatenate([data['times'] for data in ordered_dicts])
        all_values = np.concatenate([data['values'] for data in ordered_dicts])

        # Sort times and values, remove duplicates
        times, sorted_time_indices = np.unique(all_times, return_index=True)
        values = all_values[sorted_time_indices]

        # If period/freq were deferred (both None), detect period now from the combined times
        if freq_nhz is None and period_ns is None:
            nominal_period_ns = get_measure_period_ns(self.get_measure_info(measure_id)) if len(times) < 2 else None
            if nominal_period_ns is not None:
                period_ns = nominal_period_ns
            else:
                detected_period = detect_period(times)
                period_ns = int(detected_period) if not isinstance(detected_period, int) else detected_period

        time_0 = int(times[0])

        # Encode the block(s). String/object value arrays are left unforced so
        # write_data detects them and converts to int64 dictionary codes; forcing
        # a numeric raw_value_type here would trip write_data's string guard.
        if is_string_dtype(values.dtype):
            raw_v_t = None
            encoded_v_t = None
        elif np.issubdtype(values.dtype, np.integer):
            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        # encoded_time_type is left unset so write_data auto-chooses the time encoding.
        self.write_data(measure_id, device_id, times, values, freq_nhz=freq_nhz, period_ns=period_ns, time_0=time_0,
                        raw_time_type=1, raw_value_type=raw_v_t, encoded_value_type=encoded_v_t,
                        scale_m=scale_m, scale_b=scale_b, interval_index_mode="merge",
                        gap_tolerance=interval_gap_tolerance_nano, merge_blocks=merge_blocks, continuous=continuous)

    def load_device(self, device_id: int, measure_id: int|List[int] = None):
        """
        Load block metadata into RAM for a given device.

        This method loads block metadata (such as file IDs, byte ranges, and timestamps) for a
        particular device from the database and caches it locally. The caching improves the performance
        of future data queries, especially when querying the same device or measure multiple times.

        If a measure_id is specified, only blocks corresponding to that measure (or measures) will be cached.
        Otherwise, metadata for all measures associated with the device will be loaded and cached.

        :param int device_id: The device identifier. Blocks associated with this device will be fetched.
        :param int|List[int] measure_id: The measure identifier(s) associated with the metadata you want to cache.
            If None, blocks for all measures of the device will be fetched.

        """
        # Fetch block index data for the device (and measures if specified)
        block_query_result = self.sql_handler.select_blocks_for_device(device_id, measure_id)

        # Get unique file_ids
        file_id_list = list(set([row[3] for row in block_query_result]))
        if len(file_id_list) == 0:
            return
        filename_dict = self.get_filename_dict(file_id_list)

        # Build caches
        for block in block_query_result:
            block_id, measure_id, device_id, file_id, start_byte, num_bytes, start_time, end_time, num_values = block
            measure_id, device_id = int(measure_id), int(device_id)
            block = np.array([block_id, measure_id, device_id, file_id, start_byte, num_bytes, start_time, end_time, num_values], dtype=np.int64)

            append_block_cache(self.block_cache, self.start_cache, self.end_cache,
                               measure_id, device_id, block, start_time, end_time)

        freeze_block_caches(self.block_cache, self.start_cache, self.end_cache)

        # Update filename dictionary
        self.filename_dict.update(filename_dict)

    def load_definition(self, definition, gap_tolerance=None, measure_tag_match_rule="best",
                        start_time=None, end_time=None, time_units: str = "ns", cache_dir=None):
        """
        Preloads the metadata blocks for a dataset definition, then initialize device and time ranges for data fetching.
        Also preloads label data for the specified label names and devices.

        Used to speed up the iterator or many small `AtriumSDK.get_data` calls by bypassing the sql queries for data queries.

        If this method is called multiple times, the cache is overwritten with the new dataset_definition (they are not compounded).

        This method validates the provided DatasetDefinition object if its not already validated.

        :param DatasetDefinition definition: The dataset definition specifying measures, devices (or patients), and optional time ranges to include.
        :param int gap_tolerance: Tolerance for gaps between consecutive time intervals when "all" is specified in the
            definition, in the units specified by `time_units`. Defaults to 1 minute (60_000_000_000 nanoseconds).
        :param str measure_tag_match_rule: Rule for matching tags in measures; defaults to "best".
        :param int start_time: Minimum global start time for fetching data, in units of `time_units`.
        :param int end_time: Maximum global end time for fetching data, in units of `time_units`.
        :param str time_units: Time units to interpret `start_time`, `end_time`, and `gap_tolerance`.
            One of ["ns", "us", "ms", "s"]. Defaults to "ns".
        :param str cache_dir: Directory to use for caching processed blocks if caching is enabled.

        Notes:
            Supported `time_units` are nanoseconds ("ns"), microseconds ("us"), milliseconds
            ("ms"), and seconds ("s").

        Example::

            sdk = AtriumSDK(dataset_location=local_dataset_location)

            # Define measures, devices, and time ranges. load_definition takes a
            # DatasetDefinition -- not a bare dict -- so build one first.
            definition = DatasetDefinition(
                measures=["MLII"],
                device_ids={
                    1: "all",
                    2: [{"start": 1682739250000000000, "end": 1682739350000000000}],
                },
                labels=["seizure", "artifact"],
            )

            # Load the definition with time units in milliseconds
            sdk.load_definition(definition, gap_tolerance=1000, start_time=0, end_time=60000, time_units="ms")

        """
        # Validate and convert time_units
        if time_units not in time_unit_options:
            raise ValueError(f"Invalid time units. Expected one of: {list(time_unit_options.keys())}")

        # Convert start_time, end_time, and gap_tolerance to nanoseconds
        start_time_n = None if start_time is None else int(start_time * time_unit_options[time_units])
        end_time_n = None if end_time is None else int(end_time * time_unit_options[time_units])
        gap_tolerance_n = None if gap_tolerance is None else int(gap_tolerance * time_unit_options[time_units])

        # Reset the block cache and filename_dict
        self.block_cache = {}
        self.start_cache = {}
        self.end_cache = {}
        self.filename_dict = {}

        # Reset label caches
        self.label_cache = {}
        self.label_start_cache = {}
        self.label_end_cache = {}
        self.descendant_cache = {}
        self.label_lookup_caches = {}

        if not definition.is_validated:
            definition.validate(sdk=self, gap_tolerance=gap_tolerance_n,
                                measure_tag_match_rule=measure_tag_match_rule, start_time=start_time_n,
                                end_time=end_time_n)

        validated_measure_list = definition.validated_data_dict['measures']
        mapped_sources = definition.validated_data_dict['sources']
        validated_labels = definition.validated_data_dict.get('labels', [])

        # Extract measure_ids from the validated measures
        measure_ids = [measure['id'] for measure in validated_measure_list]

        # Initialize device time ranges
        device_time_ranges = defaultdict(list)

        # Process device_patient_tuples
        device_patient_tuples = mapped_sources.get('device_patient_tuples', {})
        for (device_id, _), time_ranges in device_patient_tuples.items():
            device_time_ranges[device_id].extend(time_ranges)

        # Process unmatched device_ids if any
        unmatched_device_ids = mapped_sources.get('device_ids', {})
        for device_id, time_ranges in unmatched_device_ids.items():
            device_time_ranges[device_id].extend(time_ranges)

        # Merge and sort time ranges for each device_id
        for device_id in device_time_ranges:
            time_ranges = device_time_ranges[device_id]
            # Sort time ranges
            time_ranges.sort()
            # Merge overlapping time ranges
            merged_time_ranges = []
            for start, end in sorted(time_ranges):
                if merged_time_ranges and start <= merged_time_ranges[-1][1]:
                    # Overlapping intervals, merge them
                    merged_time_ranges[-1][1] = max(merged_time_ranges[-1][1], end)
                else:
                    merged_time_ranges.append([start, end])
            device_time_ranges[device_id] = merged_time_ranges

        # Get list of device_ids
        device_ids = list(device_time_ranges.keys())

        # Fetch block index data for the devices and measures specified
        block_query_result = self.sql_handler.select_blocks_for_devices(device_ids, measure_ids)

        # Get unique file_ids
        file_id_list = list(set([row[3] for row in block_query_result]))
        if len(file_id_list) == 0:
            return
        filename_dict = self.get_filename_dict(file_id_list)

        # Build caches
        for block in block_query_result:
            block_id, measure_id, device_id, file_id, start_byte, num_bytes, block_start_time, block_end_time, num_values = block
            measure_id, device_id = int(measure_id), int(device_id)

            # Check if block's time range intersects any of the time ranges for the device_id
            device_ranges = device_time_ranges.get(device_id, [])
            if not device_ranges:
                continue  # Skip if no ranges for this device.

            if not intervals_intersect(device_ranges, block_start_time, block_end_time):
                continue  # Skip this block if no intersection.

            # Include the block
            block_array = np.array([
                block_id, measure_id, device_id, file_id, start_byte,
                num_bytes, block_start_time, block_end_time, num_values
            ], dtype=np.int64)

            append_block_cache(self.block_cache, self.start_cache, self.end_cache,
                               measure_id, device_id, block_array, block_start_time, block_end_time)

        freeze_block_caches(self.block_cache, self.start_cache, self.end_cache)

        # Update filename dictionary
        self.filename_dict.update(filename_dict)

        if validated_labels:
            self._build_label_caches(validated_labels, device_ids, device_time_ranges, start_time_n, end_time_n)

    def _build_label_caches(self, validated_labels, device_ids, device_time_ranges, global_start_time, global_end_time):
        # Get all label name IDs including descendants
        all_label_name_ids = set()
        for label_name_id in validated_labels:
            if label_name_id not in self.descendant_cache:
                descendants, ancestor_dict = collect_all_descendant_ids([label_name_id], self.sql_handler)
                self.descendant_cache[label_name_id] = (descendants, ancestor_dict)
            else:
                descendants, ancestor_dict = self.descendant_cache[label_name_id]
            all_label_name_ids.update(descendants)

        # Pre-compute lookup dictionaries for efficiency
        unique_label_set_ids = set(all_label_name_ids)
        unique_device_ids = set(device_ids)

        self.label_lookup_caches['label_set_id_to_info'] = {
            label_set_id: self.get_label_name_info(label_set_id)
            for label_set_id in unique_label_set_ids
        }
        self.label_lookup_caches['device_id_to_info'] = {
            device_id: self.get_device_info(device_id)
            for device_id in unique_device_ids
        }

        # For each device, fetch all labels within the time ranges
        for device_id in device_ids:
            device_ranges = device_time_ranges.get(device_id, [])
            if not device_ranges:
                continue

            # Initialize caches for this device
            if device_id not in self.label_cache:
                self.label_cache[device_id] = {}
                self.label_start_cache[device_id] = {}
                self.label_end_cache[device_id] = {}

            # Calculate the overall time range for this device
            device_start = min(range_start for range_start, _ in device_ranges)
            device_end = max(range_end for _, range_end in device_ranges)

            # Apply global time constraints
            if global_start_time is not None:
                device_start = max(device_start, global_start_time)
            if global_end_time is not None:
                device_end = min(device_end, global_end_time)

            # Fetch all labels for this device and time range
            labels = self.sql_handler.select_labels_with_info(
                label_set_id_list=list(all_label_name_ids),
                device_id_list=[device_id],
                start_time_n=device_start,
                end_time_n=device_end
            )

            # Group labels by label_name_id
            labels_by_name_id = defaultdict(list)
            for label_record in labels:
                label_set_id = label_record[2]
                start_time_n = label_record[7]
                end_time_n = label_record[8]

                # Check if label intersects with any device time range
                label_intersects = False
                for range_start, range_end in device_ranges:
                    if start_time_n < range_end and end_time_n > range_start:
                        label_intersects = True
                        break

                if label_intersects:
                    labels_by_name_id[label_set_id].append(label_record)

            # Cache labels for each label_name_id
            for label_name_id in all_label_name_ids:
                label_records = labels_by_name_id.get(label_name_id, [])

                if label_records:
                    # Sort by start time, then end time, then label ID
                    label_records.sort(key=lambda x: (x[7], x[8], x[0]))

                    self.label_cache[device_id][label_name_id] = label_records
                    self.label_start_cache[device_id][label_name_id] = np.array(
                        [record[7] for record in label_records], dtype=np.int64
                    )
                    self.label_end_cache[device_id][label_name_id] = np.array(
                        [record[8] for record in label_records], dtype=np.int64
                    )
                else:
                    # Initialize empty arrays for label_name_ids with no labels
                    self.label_cache[device_id][label_name_id] = []
                    self.label_start_cache[device_id][label_name_id] = np.array([], dtype=np.int64)
                    self.label_end_cache[device_id][label_name_id] = np.array([], dtype=np.int64)

    def _find_labels(self, label_name_id: int, device_id: int, start_time: int, end_time: int,
                    include_descendants: bool = True):
        """
        Find labels within the cached data that overlap with the specified time range.
        """
        if device_id not in self.label_cache:
            return []

        # Get all relevant label name IDs
        if include_descendants and label_name_id in self.descendant_cache:
            descendants, _ = self.descendant_cache[label_name_id]
            search_label_ids = descendants
        else:
            search_label_ids = {label_name_id}

        all_matching_labels = []

        for search_id in search_label_ids:
            if search_id not in self.label_cache[device_id]:
                continue

            starts = self.label_start_cache[device_id][search_id]
            ends = self.label_end_cache[device_id][search_id]

            if len(starts) == 0:
                continue

            # Find labels that overlap with the time range
            # Labels overlap if: label_start < search_end AND label_end > search_start
            end_idx = np.searchsorted(starts, end_time, side='right')
            candidate_labels = self.label_cache[device_id][search_id][:end_idx]
            candidate_ends = ends[:end_idx]

            if len(candidate_ends) > 0:
                valid_mask = candidate_ends > start_time
                matching_labels = [candidate_labels[i] for i in range(len(candidate_labels)) if valid_mask[i]]
                all_matching_labels.extend(matching_labels)

        # Sort by start time, then end time, then label ID
        all_matching_labels.sort(key=lambda x: (x[7], x[8], x[0]))
        return all_matching_labels

    def find_blocks(self, measure_id: int, device_id: int, start_time: int, end_time: int):
        """
        Find blocks within the cached data that overlap with the specified time range.
        """
        if measure_id not in self.block_cache or device_id not in self.block_cache[measure_id]:
            return []

        blocks = self.block_cache[measure_id][device_id]
        starts = self.start_cache[measure_id][device_id]
        ends = self.end_cache[measure_id][device_id]

        end_idx = np.searchsorted(starts, end_time, side='right')
        candidate_blocks = blocks[:end_idx]
        candidate_ends = ends[:end_idx]

        valid_mask = candidate_ends > start_time
        return candidate_blocks[valid_mask]

    # ------------------------------------------------------------------ #
    # Measures
    # ------------------------------------------------------------------ #
    def get_measure_id(self, measure_tag: str, freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                       period: Union[int, float] = None, time_units: str = None):
        """
        .. _get_measure_id_label:

        Returns the identifier for a measure specified by its tag, frequency or period, units, and frequency/time units.

        :param str measure_tag: The tag of the measure.
        :param float freq: The frequency of the measure (mutually exclusive with period).
        :param str units: The unit of the measure (default is an empty string).
        :param str freq_units: The frequency unit of the measure (default is 'nHz').
        :param float period: The period of the measure (mutually exclusive with freq).
        :param str time_units: The time unit for the period (default is 'ns').
        :return: The identifier of the measure.
        :rtype: int

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> measure_tag = "Temperature Measure"
        >>> freq = 100.0
        >>> units = "Celsius"
        >>> freq_units = "Hz"
        >>> sdk.get_measure_id(measure_tag, freq=freq, units=units, freq_units=freq_units)
        ... 7
        >>> # Using period instead
        >>> sdk.get_measure_id(measure_tag, period=0.01, time_units="s", units=units)
        ... 7
        >>> measure_tag = "Measure That Does Not Exist."
        >>> sdk.get_measure_id(measure_tag, freq=freq, units=units, freq_units=freq_units)
        ... None
        """
        # Check for mutually exclusive parameters
        if freq is not None and period is not None:
            raise ValueError("freq and period are mutually exclusive. Specify only one.")

        if freq is None and period is None:
            raise ValueError("Either freq or period must be specified.")

        # Set default values for units and freq_units/time_units if not provided
        units = "" if units is None else units
        freq_units = "nHz" if freq_units is None else freq_units
        time_units = "ns" if time_units is None else time_units

        # Convert to nanohertz based on which parameter was provided
        if freq is not None:
            freq_nhz = convert_to_nanohz(freq, freq_units)
        else:  # period is not None
            period_ns = int(period * time_unit_options[time_units])
            freq_nhz = 10 ** 18 // period_ns

        # Force python int
        freq_nhz = int(freq_nhz)

        # If metadata connection type is "api", use API method to get the measure ID
        if self.metadata_connection_type == "api":
            return self._api_get_measure_id(measure_tag, freq_nhz, units, "nHz")

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
            (in nanohertz), period (in nanoseconds), code, unit, unit label, unit code, source_id, and the measure-kind
            fields ``signal_kind`` and ``value_type``. ``signal_kind`` is the temporal shape of the signal
            (one of ``waveform | sample | event | state``) and ``value_type`` is the value encoding
            (``numeric | string``). Both are resolved with read-time defaults: a measure stored without them reads
            back as ``waveform`` / ``numeric`` (a ``value_type`` that was never set but has string data written to it
            resolves to ``string``).
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
            'period_ns': 1000000000,
            'code': 'HR',
            'unit': 'BPM',
            'unit_label': 'beats per minute',
            'unit_code': 264864,
            'source_id': 1,
            'signal_kind': 'waveform',
            'value_type': 'numeric'
        }

        """
        # Check if metadata connection type is API
        if self.metadata_connection_type == "api":
            measure_info = self._request("GET", f"measures/{measure_id}")
            if measure_info:
                # Add period_ns to API response if not present
                if 'period_ns' not in measure_info or measure_info['period_ns'] is None:
                    measure_info['period_ns'] = 10 ** 18 // measure_info['freq_nhz']
            return measure_info

        # If measure_id is already in the cache, return the cached measure info
        if measure_id in self._measures:
            return self._measures[measure_id]

        # Query the SQL database for the measure information
        row = self.sql_handler.select_measure(measure_id=measure_id)

        # If no row is found, return None
        if row is None:
            return None

        # Unpack the row tuple into variables (includes period_ns and the
        # signal_kind / value_type columns).
        measure_id, measure_tag, measure_name, measure_freq_nhz, stored_period_ns, measure_code, measure_unit, \
            measure_unit_label, measure_unit_code, measure_source_id, stored_signal_kind, stored_value_type = row

        # Use stored period_ns if available, otherwise calculate from freq_nhz
        if stored_period_ns is not None:
            measure_period_ns = stored_period_ns
        else:
            measure_period_ns = 10 ** 18 // measure_freq_nhz

        signal_kind, value_type = self._resolve_measure_kind(
            measure_id, stored_signal_kind, stored_value_type)

        # Create a dictionary containing the measure information
        measure_info = {
            'id': measure_id,
            'tag': measure_tag,
            'name': measure_name,
            'freq_nhz': measure_freq_nhz,
            'period_ns': measure_period_ns,
            'code': measure_code,
            'unit': measure_unit,
            'unit_label': measure_unit_label,
            'unit_code': measure_unit_code,
            'source_id': measure_source_id,
            'signal_kind': signal_kind,
            'value_type': value_type
        }

        # Cache the measure information in the _measures dictionary
        self._measures[measure_id] = measure_info

        # Return the measure information dictionary
        return measure_info

    def search_measures(self, tag_match=None, freq=None, unit=None, name_match=None, freq_units=None,
                        period=None, time_units=None):
        """
        .. _search_measures_label:

        Retrieve information about all measures in the linked relational database that match the specified search criteria.

        This function filters the measures based on the provided search criteria and returns a dictionary containing
        information about each matching measure, including its id, tag, name, sample frequency (in nanohertz),
        period (in nanoseconds), code, unit, unit label, unit code, and source_id.

        :param tag_match: A string to match against the `measure_tag` field. If not None, only measures with a `measure_tag`
            field containing this string will be returned.
        :type tag_match: str, optional
        :param freq: A value to match against the `measure_freq_nhz` field. If not None, only measures with a
            `measure_freq_nhz` field equal to this value will be returned. Mutually exclusive with period.
        :type freq: int, optional
        :param unit: A string to match against the `measure_unit` field. If not None, only measures with a `measure_unit`
            field equal to this string will be returned.
        :type unit: str, optional
        :param name_match: A string to match against the `measure_name` field. If not None, only measures with a
            `measure_name` field containing this string will be returned.
        :type name_match: str, optional
        :param freq_units: The units for the freq parameter. (Default: "Hz")
        :type freq_units: str, optional
        :param period: A value to match against the period. If not None, only measures with a matching period will be returned.
            Mutually exclusive with freq.
        :type period: float, optional
        :param time_units: The units for the period parameter. (Default: "ns")
        :type time_units: str, optional
        :return: A dictionary containing information about each measure that matches the specified search criteria.
        :rtype: dict

        Every criterion is independently optional and they combine with AND. Calling with no
        arguments at all returns every measure::

            sdk.search_measures(tag_match="heart-rate")        # by tag substring
            sdk.search_measures(unit="mmHg")                   # by unit alone
            sdk.search_measures(name_match="Arterial")         # by name substring
            sdk.search_measures(freq=60, freq_units="Hz")      # by frequency
            sdk.search_measures()                              # everything

        .. note::

            **This method cannot filter on** ``signal_kind`` **or** ``value_type``. To find,
            say, every string measure, pull the lot and filter client-side::

                strings = {mid: info for mid, info in sdk.get_all_measures().items()
                           if info["value_type"] == "string"}

            See :ref:`Measure Metadata <measure_metadata>` for what those two fields mean.
        """
        # Check for mutually exclusive parameters
        if freq is not None and period is not None:
            raise ValueError("freq and period are mutually exclusive. Specify only one.")

        # Check the metadata connection type and call the appropriate API search method if necessary
        if self.metadata_connection_type == "api":
            params = {'measure_tag': tag_match, 'freq': freq, 'unit': unit, 'measure_name': name_match,
                      'freq_units': freq_units}
            result = self._request("GET", "measures/", params=params)
            # Add period_ns to each measure in API response
            for measure_id, measure_info in result.items():
                measure_info['period_ns'] = 10 ** 18 // measure_info['freq_nhz']
            return result

        # Set the default frequency units to "Hz" if not provided
        freq_units = "Hz" if freq_units is None else freq_units
        time_units = "ns" if time_units is None else time_units

        # Convert the frequency or period to nanohertz if necessary
        target_freq_nhz = None
        if freq is not None and freq_units != "nHz":
            target_freq_nhz = convert_to_nanohz(freq, freq_units)
        elif freq is not None:
            target_freq_nhz = freq
        elif period is not None:
            period_ns = int(period * time_unit_options[time_units])
            target_freq_nhz = 10 ** 18 // period_ns
        # Get all measures from the database
        all_measures = self.get_all_measures()

        # Initialize the result dictionary
        result = {}

        # Iterate through all measures and filter them based on the search criteria
        for measure_id, measure_info in all_measures.items():
            # Create a list of boolean values for each search criterion
            match_bool_list = [
                tag_match is None or tag_match in measure_info['tag'],
                target_freq_nhz is None or target_freq_nhz == measure_info['freq_nhz'],
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
             'freq_nhz': 500000000000,
             'period_ns': 2000000,
             'code': 'HR',
             'unit': 'BPM',
             'unit_label': 'Beats per Minute',
             'unit_code': 'BPM',
             'source_id': 1},
         2: {'id': 2,
             'tag': 'Respiration Rate',
             'name': 'Respiration Rate Measurement',
             'freq_nhz': 500000000000,
             'period_ns': 2000000,
             'code': 'RR',
             'unit': 'BPM',
             'unit_label': 'Breaths per Minute',
             'unit_code': 'BPM',
             'source_id': 1}}

        :return: A dictionary containing information about each measure, including its id, tag, name, sample frequency
            (in nanohertz), period (in nanoseconds), code, unit, unit label, unit code, and source_id.
        :rtype: dict
        """
        # Check if connection type is API and call the appropriate method
        if self.metadata_connection_type == "api":
            measure_dict = self._request("GET", "measures/")
            result = {int(measure_id): measure_info for measure_id, measure_info in measure_dict.items()}
            # Add period_ns to each measure in API response if not present
            for measure_id, measure_info in result.items():
                if 'period_ns' not in measure_info or measure_info['period_ns'] is None:
                    measure_info['period_ns'] = 10 ** 18 // measure_info['freq_nhz']
            return result

        # Get all measures from the SQL handler
        measure_tuple_list = self.sql_handler.select_all_measures()

        # Initialize an empty dictionary to store measure information
        measure_dict = {}

        # Iterate through the list of measures and construct a dictionary for each measure
        for measure_info in measure_tuple_list:
            measure_id, measure_tag, measure_name, measure_freq_nhz, stored_period_ns, measure_code, \
                measure_unit, measure_unit_label, measure_unit_code, measure_source_id, \
                stored_signal_kind, stored_value_type = measure_info

            # Use stored period_ns if available, otherwise calculate from freq_nhz
            if stored_period_ns is not None:
                measure_period_ns = stored_period_ns
            elif measure_freq_nhz:
                measure_period_ns = 10 ** 18 // measure_freq_nhz
            else:
                measure_period_ns = None
                _LOGGER.warning(
                    "Measure %d (%s) has no usable frequency, the legacy marker for an "
                    "aperiodic signal. Run AtriumSDK(auto_upgrade=True) to convert it to "
                    "a declared aperiodic sample measure.", measure_id, measure_tag)

            signal_kind, value_type = self._resolve_measure_kind(
                measure_id, stored_signal_kind, stored_value_type)

            # Add the measure information to the dictionary
            measure_dict[measure_id] = {
                'id': measure_id,
                'tag': measure_tag,
                'name': measure_name,
                'freq_nhz': measure_freq_nhz,
                'period_ns': measure_period_ns,
                'code': measure_code,
                'unit': measure_unit,
                'unit_label': measure_unit_label,
                'unit_code': measure_unit_code,
                'source_id': measure_source_id,
                'signal_kind': signal_kind,
                'value_type': value_type
            }

        return measure_dict

    def get_measure_id_list_from_tag(self, measure_tag: str, approx=True, freq=None, units=None, freq_units=None,
                                     period=None, time_units=None):
        """
        Returns a list of matching measure_ids for a given tag in DESC order by number of stored blocks.
        Helpful for finding all ids or the most prevalent id for a given tag. Optionally filters by frequency/period and units.

        :param str measure_tag: The tag of the measure.
        :param bool approx: If True, approximates the result based on first 100,000 rows of the block table.
            If False, queries the entire block table.
        :param freq: Optional frequency to filter measures. Mutually exclusive with period.
        :param units: Optional units of the measure to filter by.
        :param freq_units: Units of the provided frequency. Converts frequency to nanohertz if not already.
        :param period: Optional period to filter measures. Mutually exclusive with freq.
        :param time_units: Units of the provided period. (Default is 'ns')
        :return: A list of measure_ids
        """
        # Check for mutually exclusive parameters
        if freq is not None and period is not None:
            raise ValueError("freq and period are mutually exclusive. Specify only one.")

        # Convert frequency or period to nanohertz if necessary
        target_freq_nhz = None
        if freq is not None and freq_units and freq_units != "nHz":
            target_freq_nhz = convert_to_nanohz(freq, freq_units)
        elif freq is not None:
            target_freq_nhz = freq
        elif period is not None:
            time_units = "ns" if time_units is None else time_units
            period_ns = int(period * time_unit_options[time_units])
            target_freq_nhz = 10 ** 18 // period_ns

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        # Get initial list of measure IDs from tag
        measure_ids = self._measure_tag_to_ordered_id.get(measure_tag, [])
        if not measure_ids:
            # Reload the cache if not found
            self._measure_tag_to_ordered_id = self.sql_handler.get_tag_to_measure_ids_dict(approx=approx)
            measure_ids = self._measure_tag_to_ordered_id.get(measure_tag, [])

        # Filter measure_ids by frequency and units if necessary
        if target_freq_nhz is not None or units is not None:
            filtered_measure_ids = []
            for measure_id in measure_ids:
                measure_info = self.get_measure_info(measure_id)
                if target_freq_nhz is not None and measure_info.get('freq_nhz') != target_freq_nhz:
                    continue
                if units is not None and measure_info.get('unit') != units:
                    continue
                filtered_measure_ids.append(measure_id)
            measure_ids = filtered_measure_ids

        return measure_ids

    def insert_measure(self, measure_tag: str, freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                       period: Union[int, float] = None, time_units: str = None, measure_name: str = None,
                       measure_id: int = None, code: str = None, unit_label: str = None,
                       unit_code: str = None, source_id: int = None, source_name: str = None,
                       signal_kind: str = None, value_type: str = None):
        """
        .. _insert_measure_label:

        Defines a new signal type to be stored in the dataset, as well as defining metadata related to the signal.

        `measure_tag`, and either `freq` or `period`, and `units` are required information.

        >>> # Define a new signal with frequency and additional metadata.
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
        >>>
        >>> # Define a new signal with period instead of frequency
        >>> period = 2  # 2 milliseconds
        >>> time_units = "ms"
        >>> measure_tag = "ECG Lead II - 2ms period"
        >>> measure_id = sdk.insert_measure(measure_tag=measure_tag, period=period, time_units=time_units,
                                            units=units, measure_name=measure_name)

        :param str measure_tag: A short string identifying the signal.
        :param freq: The sample frequency of the signal. Mutually exclusive with period.
        :param str units: The units of the signal.
        :param str freq_units: The unit used for the specified frequency. This value can be one of ["Hz",
            "kHz", "MHz"]. Keep in mind if you use extremely large values for this it will be
            converted to nano hertz in the backend, and you may overflow 64bit integers. Default is nano hertz.
        :param period: The sample period of the signal. Mutually exclusive with freq.
        :param str time_units: The unit used for the specified period. This value can be one of ["s", "ms", "us", "ns"].
            Default is nanoseconds.
        :param str measure_name: A long form description of the signal (optional).
        :param int measure_id: The desired measure_id (optional).
        :param str code: A specific code identifying the signal (optional).
        :param str unit_label: A label for the unit (optional).
        :param str unit_code: A code for the unit (optional).
        :param int source_id: An identifier for the data source (optional).
        :param str source_name: The name of the data source associated with the measure, used if source_id is not
            provided (optional).
        :param str signal_kind: Optional temporal shape of the signal, one of
            ``waveform | sample | event | state``. When omitted the measure defaults to
            ``waveform`` at read time; ``sample`` is the safe default for aperiodic
            numeric data. The shape is never inferred beyond ``waveform`` -- pass this
            hint explicitly for sample/event/state measures.
        :param str value_type: Optional value encoding, one of ``numeric | string``.
            When omitted it is inferred at first write from the value dtype
            (string/object -> ``string``, else ``numeric``); an explicit value wins.

        :return: The measure_id of the inserted or existing measure. Always a real id
            (>= 1) -- if a concurrent writer wins the race to create this measure, the
            existing row's id is read back and returned rather than the driver's empty
            ``lastrowid``.
        :rtype: int
        :raises RuntimeError: If no id could be obtained -- the insert reported no new
            row and no matching row could be read back.

        """

        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        # Validate the measure-kind enums when explicitly provided.
        validate_measure_kind_values(signal_kind, value_type)

        # Reject the one invalid combination at the point the mistake is made rather
        # than hours later inside get_iterator's fill path. A string measure whose
        # signal_kind is (or defaults to) 'waveform' is auto-corrected to 'event' and
        # the caller is told how to choose 'state'/'sample' instead. Auto-correction
        # rather than a raise, because this is also the shape a legacy dataset carries
        # into transfer_measures -> insert_measure, and a transfer of an already-broken
        # source measure must repair it, not abort.
        if is_invalid_kind_combination(signal_kind, value_type):
            _LOGGER.warning(invalid_kind_combination_message(
                f"'{measure_tag}'", STRING_SIGNAL_KIND_FALLBACK, measure_exists=False))
            signal_kind = STRING_SIGNAL_KIND_FALLBACK

        # Check for mutually exclusive parameters
        if freq is not None and period is not None:
            raise ValueError("freq and period are mutually exclusive. Specify only one.")

        if freq is None and period is None:
            raise ValueError("Either freq or period must be specified.")

        # Validate freq / period BEFORE any row is written. Inserting the measure row
        # first and only THEN computing ``10 ** 18 // freq_nhz`` would commit a row with
        # freq_nhz=0, after which every later ``AtriumSDK(...)`` on that dataset dies in
        # get_all_measures on the same expression -- the dataset could never be opened
        # again. freq=0 is exactly what an aperiodic ingest reaches for, so the message
        # points at the supported route.
        if freq is not None and freq <= 0:
            raise ValueError(
                f"freq must be greater than 0; got {freq!r}. The frequency is used as a divisor "
                f"(period_ns = 10**18 // freq_nhz), so 0 is not a usable value and a negative one "
                f"makes every raster computation meaningless. Do NOT use freq=0 to mean "
                f"'aperiodic': give the measure a nominal frequency (e.g. freq=1, freq_units='Hz') "
                f"and declare its temporal shape with signal_kind='sample' | 'event' | 'state', "
                f"then write it with write_time_value_pairs().")
        if period is not None and period <= 0:
            raise ValueError(
                f"period must be greater than 0; got {period!r}. Do NOT use period=0 to mean "
                f"'aperiodic': give the measure a nominal period and declare its temporal shape "
                f"with signal_kind='sample' | 'event' | 'state', then write it with "
                f"write_time_value_pairs().")

        # Check if measure_tag, measure_name, and units are either strings or None
        assert isinstance(measure_tag, str)
        assert isinstance(measure_name, str) or measure_name is None
        assert isinstance(units, str) or units is None
        assert isinstance(code, str) or code is None
        assert isinstance(unit_label, str) or unit_label is None
        assert isinstance(unit_code, str) or unit_code is None

        # Set default frequency/time units if not provided
        freq_units = "nHz" if freq_units is None else freq_units
        time_units = "ns" if time_units is None else time_units
        units = "" if units is None else units

        # Handle source_name to source_id conversion
        if source_name and not source_id:
            source_id = self.get_source_id(source_name)
            if source_id is None:
                raise ValueError(f"Source name {source_name} not found.")

        # Convert to nanohertz based on which parameter was provided
        if freq is not None:
            # Convert frequency to nanohertz if the provided frequency unit is not "nHz"
            if freq_units != "nHz":
                freq_nhz = convert_to_nanohz(freq, freq_units)
            else:
                freq_nhz = freq
            period_ns = None  # Will be calculated from freq_nhz later
        else:  # period is not None
            # Convert period to nanoseconds then to equivalent frequency in nanohertz
            period_ns = int(period * time_unit_options[time_units])
            if period_ns <= 0:
                raise ValueError(
                    f"period {period!r} {time_units} rounds to {period_ns} nanoseconds, which is "
                    f"not a usable sampling period. Specify a period of at least 1 nanosecond.")
            freq_nhz = 10 ** 18 // period_ns

        # Force Cast Python integer
        freq_nhz = int(freq_nhz)

        # A frequency that rounds down to zero would commit the same unopenable
        # freq_nhz=0 row as freq=0 itself, so it is rejected here too, still before
        # any insert.
        if freq_nhz <= 0:
            raise ValueError(
                f"freq {freq!r} {freq_units} converts to {freq_nhz} nHz, which is not a usable "
                f"frequency (it is used as a divisor: period_ns = 10**18 // freq_nhz). Specify a "
                f"frequency of at least 1 nHz.")

        # Check for id clash
        if measure_id is not None:
            assert isinstance(measure_id, int)
            measure_id = int(measure_id)
            measure_info = self.get_measure_info(measure_id)
            if measure_info is not None:
                if measure_info['tag'] == measure_tag and \
                        measure_info['freq_nhz'] == freq_nhz and \
                        measure_info['unit'] == units:
                    self._apply_kind_to_existing_measure(measure_id, signal_kind, value_type)
                    return measure_id
                raise ValueError(f"Inserted measure_id {measure_id} already exists with data: {measure_info}")

        # Check if the measure already exists in the dataset
        check_measure_id = self.get_measure_id(measure_tag, freq=freq_nhz, freq_units="nHz", units=units)
        if check_measure_id is not None:
            self._apply_kind_to_existing_measure(check_measure_id, signal_kind, value_type)
            return check_measure_id

        # Insert the new measure into the database
        inserted_measure_id = self.sql_handler.insert_measure(
            measure_tag, freq_nhz, units, measure_name, measure_id=measure_id, code=code, unit_label=unit_label,
            unit_code=unit_code, source_id=source_id, period_ns=period_ns, signal_kind=signal_kind,
            value_type=value_type)

        if inserted_measure_id is None or int(inserted_measure_id) <= 0:
            resolved_measure_id = self.get_measure_id(
                measure_tag, freq=freq_nhz, freq_units="nHz", units=units)
            if resolved_measure_id is None or int(resolved_measure_id) <= 0:
                raise RuntimeError(
                    f"insert_measure could not obtain a measure_id for "
                    f"(tag={measure_tag!r}, freq_nhz={freq_nhz}, units={units!r}): the insert "
                    f"reported no new row and no matching row could be read back. This is "
                    f"normally a lost get-or-insert race whose winner is re-read on the next "
                    f"line, so a failure here means the row is genuinely absent -- check that "
                    f"the metadata database is reachable and writable, and that nothing rolled "
                    f"the insert back.")
            resolved_measure_id = int(resolved_measure_id)
            self._apply_kind_to_existing_measure(resolved_measure_id, signal_kind, value_type)
            return resolved_measure_id

        inserted_measure_id = int(inserted_measure_id)

        # Calculate period_ns from freq_nhz for cache (if not already calculated)
        if period_ns is None:
            period_ns = 10 ** 18 // freq_nhz

        # Add new measure_id into cache. Apply the same read-time defaulting as
        # get_measure_info so the cached view matches a fresh DB read (a NULL
        # value_type with no dictionary file yet reads as 'numeric').
        cached_signal_kind, cached_value_type = self._resolve_measure_kind(
            inserted_measure_id, signal_kind, value_type)
        measure_info = {
            'id': inserted_measure_id,
            'tag': measure_tag,
            'name': measure_name,
            'freq_nhz': freq_nhz,
            'period_ns': period_ns,
            'code': code,
            'unit': units,
            'unit_label': unit_label,
            'unit_code': unit_code,
            'source_id': source_id,
            'signal_kind': cached_signal_kind,
            'value_type': cached_value_type
        }
        self._measure_ids[(measure_tag, freq_nhz, units)] = inserted_measure_id
        self._measures[inserted_measure_id] = measure_info

        return inserted_measure_id

    def get_or_insert_measure(self, measure_tag: str, freq: Union[int, float] = None, units: str = None,
                              freq_units: str = None, period: Union[int, float] = None, time_units: str = None,
                              measure_name: str = None, code: str = None, unit_label: str = None,
                              unit_code: str = None, source_id: int = None, source_name: str = None,
                              signal_kind: str = None, value_type: str = None) -> int:
        """
        .. _get_or_insert_measure_label:

        Return the measure_id for ``(measure_tag, freq, units)``, creating the measure if
        it does not exist yet.

        This is the idiom every ingest pipeline needs on its first sight of a signal, and
        which is otherwise hand-rolled as a :meth:`get_measure_id` followed by an
        :meth:`insert_measure` on the miss.
        :meth:`insert_measure` has always been a get-or-insert; this is the name that says
        so, so a caller does not have to read its body to find that out. The return value
        is always a real measure_id -- never ``None``, never ``0`` -- including when a
        concurrent process wins the race to create the same measure.

        Parameters mean exactly what they mean on :meth:`insert_measure`, minus
        ``measure_id``: requesting a specific id is an "insert this exact row" operation,
        not a get-or-insert, so it is left on :meth:`insert_measure` where it belongs.

        ``signal_kind`` / ``value_type``, when given, are also applied to a measure that
        already exists (with a warning naming the change), so an ingest pipeline can
        classify a measure that an earlier run created without that metadata.

        >>> # Safe to call on every message; creates the measure once.
        >>> measure_id = sdk.get_or_insert_measure(
        ...     measure_tag="alarm_text", freq=1, freq_units="Hz", units="string",
        ...     signal_kind="event", value_type="string")

        :param str measure_tag: A short string identifying the signal.
        :param freq: The sample frequency of the signal. Mutually exclusive with period.
        :param str units: The units of the signal.
        :param str freq_units: The unit used for the specified frequency, one of ["Hz",
            "kHz", "MHz"]. Default is nano hertz.
        :param period: The sample period of the signal. Mutually exclusive with freq.
        :param str time_units: The unit used for the specified period, one of
            ["s", "ms", "us", "ns"]. Default is nanoseconds.
        :param str measure_name: A long form description of the signal (optional).
        :param str code: A specific code identifying the signal (optional).
        :param str unit_label: A label for the unit (optional).
        :param str unit_code: A code for the unit (optional).
        :param int source_id: An identifier for the data source (optional).
        :param str source_name: The name of the data source, used if source_id is not
            provided (optional).
        :param str signal_kind: Optional temporal shape, one of
            ``waveform | sample | event | state``. See :meth:`insert_measure`.
        :param str value_type: Optional value encoding, one of ``numeric | string``.
            See :meth:`insert_measure`.

        :return: The measure_id of the existing or newly created measure.
        :rtype: int
        :raises RuntimeError: If no id could be obtained (see :meth:`insert_measure`).
        """
        return self.insert_measure(
            measure_tag=measure_tag, freq=freq, units=units, freq_units=freq_units, period=period,
            time_units=time_units, measure_name=measure_name, code=code, unit_label=unit_label,
            unit_code=unit_code, source_id=source_id, source_name=source_name,
            signal_kind=signal_kind, value_type=value_type)

    # ------------------------------------------------------------------ #
    # Devices
    # ------------------------------------------------------------------ #
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

        :return: The device_id of the inserted or existing device. Always a real id
            (>= 1) -- if a concurrent writer wins the race to create this device, the
            existing row's id is read back and returned rather than the driver's empty
            ``lastrowid``.
        :rtype: int

        Raises:
            ValueError: If specified device_id already exists with a different device_tag.
                        If bed_name or source_name is provided but does not match any existing records.
            RuntimeError: If no id could be obtained -- the insert reported no new row and
                        no matching row could be read back.
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
            device_id = int(device_id)
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
        inserted_device_id = self.sql_handler.insert_device(device_tag, device_name, device_id, manufacturer, model,
                                                            device_type, bed_id, source_id)

        if inserted_device_id is None or int(inserted_device_id) <= 0:
            resolved_device_id = self.get_device_id(device_tag)
            if resolved_device_id is None or int(resolved_device_id) <= 0:
                raise RuntimeError(
                    f"insert_device could not obtain a device_id for device_tag "
                    f"{device_tag!r}: the insert reported no new row and no matching row "
                    f"could be read back. This is normally a lost get-or-insert race whose "
                    f"winner is re-read on the next line, so a failure here means the row is "
                    f"genuinely absent -- check that the metadata database is reachable and "
                    f"writable, and that nothing rolled the insert back.")
            return int(resolved_device_id)

        return int(inserted_device_id)

    def get_or_insert_device(self, device_tag: str, device_name: str = None, manufacturer: str = None,
                             model: str = None, device_type: str = None, bed_id: int = None,
                             bed_name: str = None, source_id: int = None, source_name: str = None) -> int:
        """
        .. _get_or_insert_device_label:

        Return the device_id for ``device_tag``, creating the device if it does not exist
        yet.

        The device twin of :meth:`get_or_insert_measure`: the call an ingest pipeline
        makes on its first sight of a device, instead of hand-rolling a
        :meth:`get_device_id` followed by an :meth:`insert_device` on the miss. The
        return value is always a real
        device_id -- never ``None``, never ``0`` -- including when a concurrent process
        wins the race to create the same device.

        Parameters mean exactly what they mean on :meth:`insert_device`, minus
        ``device_id``: requesting a specific id is an "insert this exact row" operation,
        not a get-or-insert. Metadata (``device_name``, ``manufacturer``, ...) is used
        only when the device is actually created; an existing device is returned
        unchanged.

        >>> # Safe to call on every message; creates the device once.
        >>> device_id = sdk.get_or_insert_device(device_tag="Monitor A3")

        :param str device_tag: A unique string identifying the device (required).
        :param str device_name: A long form description of the device (optional).
        :param str manufacturer: The device's manufacturer (optional).
        :param str model: The device's model (optional).
        :param str device_type: The type of the device, 'static' or 'dynamic' (optional).
        :param int bed_id: The ID of the bed associated with the device (optional).
        :param str bed_name: The name of the bed, used if bed_id is not provided (optional).
        :param int source_id: The ID of the data source (optional).
        :param str source_name: The name of the data source, used if source_id is not
            provided (optional).

        :return: The device_id of the existing or newly created device.
        :rtype: int
        :raises RuntimeError: If no id could be obtained (see :meth:`insert_device`).
        """
        return self.insert_device(
            device_tag=device_tag, device_name=device_name, manufacturer=manufacturer, model=model,
            device_type=device_type, bed_id=bed_id, bed_name=bed_name, source_id=source_id,
            source_name=source_name)

    # ------------------------------------------------------------------ #
    # Patients and patient history
    # ------------------------------------------------------------------ #
    def get_patient_id(self, mrn: str):
        """
        Retrieve the patient ID associated with a given medical record number (MRN).

        This method looks for a patient's ID using their MRN. If the patient ID is not found in the initial search,
        it triggers a refresh of all patient data and searches again.

        :param str mrn: The medical record number for the patient. An int can be provided, but will be converted and stored as a string.
        :return: The patient ID as an integer if the patient is found; otherwise, None.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> patient_id = sdk.get_patient_id(mrn="123456")
        >>> print(patient_id)
        1
        """

        # Convert mrn to str for lookup
        mrn = str(mrn)

        # Check if we are in API mode
        if self.metadata_connection_type == "api":
            return self._request("GET", f"/patients/mrn|{mrn}", params={'time': None})['id']

        if mrn in self._mrn_to_patient_id:
            return self._mrn_to_patient_id[mrn]

        self.get_all_patients()

        if mrn in self._mrn_to_patient_id:
            return self._mrn_to_patient_id[mrn]

        return None

    def get_mrn(self, patient_id: int):
        """
        Retrieve the medical record number (MRN) associated with a given patient ID.

        This method searches for a patient's MRN using their patient ID. If the MRN is not found in the initial search,
        it triggers a refresh of all patient data and searches again.

        :param patient_id: The numeric identifier for the patient.
        :return: The MRN as a string if the patient is found; otherwise, None.

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> mrn = sdk.get_mrn(patient_id=1)
        >>> print(mrn)
        '123456'
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

    def get_patient_info(self, patient_id: int = None, mrn: str = None, time: int = None, time_units: str = None):
        """
        Retrieve information about a specific patient using either their numeric patient id or medical record number (MRN).

        :param int patient_id: The numeric identifier for the patient.
        :param str mrn: The medical record number for the patient. An int can be provided, but will be converted and stored as a string.
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
            'mrn': '123456',
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
            # Convert mrn to str for proper lookup
            mrn = str(mrn)
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
             'mrn': '123456',
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
             'mrn': '654321',
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

    def get_mrn_to_patient_id_map(self, mrn_list: List[str] = None):
        """
        Get a mapping of Medical Record Numbers (MRNs) to patient IDs.

        This method queries the metadata store for all patients with MRNs in the given list
        and returns a dictionary with MRNs as keys and patient IDs as values. MRNs that
        cannot be found are omitted from the result.

        :param mrn_list: A list of MRNs to filter the patients, or None to get all patients. Int values can be provided, but will be converted and stored as strings.
        :type mrn_list: List[str], optional
        :return: A dictionary with MRNs (as strings) as keys and patient IDs as values.
        :rtype: dict
        """
        if self.metadata_connection_type == "api":
            if mrn_list is None:
                # Fetch every patient from the API and build the map
                all_patients = self._api_get_all_patients()
                return {
                    str(info['mrn']): int(patient_id)
                    for patient_id, info in all_patients.items()
                    if info.get('mrn') is not None
                }

            if not mrn_list:
                return {}

            result_dict = {}
            for mrn in mrn_list:
                try:
                    result_temp = self._request("GET", f"patients/mrn|{mrn}", params={'time': None})
                except ValueError:
                    # MRN not found on the server; skip it to match SQL-branch behavior
                    continue
                if result_temp and result_temp.get('id') is not None:
                    result_dict[str(mrn)] = int(result_temp['id'])
            return result_dict

        # If mrn_list is None, get all patients
        if mrn_list is None:
            # refresh cache
            self.get_all_patients()
            return dict(self._mrn_to_patient_id)

        # Convert all mrns to strings for consistent lookup
        mrn_list_str = [str(mrn) for mrn in mrn_list]

        # If all mrns are in the cache
        if all(m in self._mrn_to_patient_id for m in mrn_list_str):
            return {m: self._mrn_to_patient_id[m] for m in mrn_list_str}

        # Refresh the cache and return all available mrns.
        self.get_all_patients()
        return {m: self._mrn_to_patient_id[m] for m in mrn_list_str if m in self._mrn_to_patient_id}

    def get_patient_id_to_mrn_map(self, patient_id_list: List[int] = None):
        """
        Get a mapping of patient IDs to Medical Record Numbers (MRNs).

        This method queries the metadata store for all patients with IDs in the given list
        and returns a dictionary with patient IDs as keys and MRNs as values. Patients whose
        MRN is not set, or who cannot be found, are omitted from the result.

        :param patient_id_list: A list of patient IDs to filter the patients, or None to get all patients.
        :type patient_id_list: List[int], optional
        :return: A dictionary with patient IDs (as ints) as keys and MRNs (as strings) as values.
        :rtype: dict
        """
        if self.metadata_connection_type == "api":
            if patient_id_list is None:
                all_patients = self._api_get_all_patients()
                return {
                    int(pid): str(info['mrn'])
                    for pid, info in all_patients.items()
                    if info.get('mrn') is not None
                }

            if not patient_id_list:
                return {}

            result_dict = {}
            for patient_id in patient_id_list:
                try:
                    result_temp = self._request("GET", f"patients/id|{patient_id}", params={'time': None})
                except ValueError:
                    # Patient not found on the server; skip it
                    continue
                if result_temp and result_temp.get('mrn') is not None:
                    result_dict[int(patient_id)] = str(result_temp['mrn'])
            return result_dict

        # If patient_id_list is None, get all patients
        if patient_id_list is None:
            # refresh cache
            self.get_all_patients()
            return dict(self._patient_id_to_mrn)

        # Coerce all ids to int for consistent lookup
        patient_id_list_int = [int(pid) for pid in patient_id_list]

        # If all ids are in the cache
        if all(pid in self._patient_id_to_mrn for pid in patient_id_list_int):
            return {pid: self._patient_id_to_mrn[pid] for pid in patient_id_list_int}

        # Refresh the cache and return all available ids.
        self.get_all_patients()
        return {pid: self._patient_id_to_mrn[pid] for pid in patient_id_list_int if pid in self._patient_id_to_mrn}

    def insert_patient(self, patient_id: int = None, mrn: str = None, gender: str = None, dob: int = None,
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
        :param str mrn: The Medical Record Number (MRN) of the patient. An int can be provided, but will be converted and stored as a string.
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

        # Convert mrn to string if provided
        if mrn is not None:
            mrn = str(mrn)

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

    def get_patient_history(self, patient_id: int = None, mrn: str = None, field: str = None, start_time: int = None,
                            end_time: int = None, time_units: str = None):
        """
        Retrieve a list of a patients historical measurements using either their numeric patient id or medical record number (MRN).
        If start_time and end_time are left empty it will give all the patient's history. The results are returned in ascending order by time.

        :param int patient_id: The numeric identifier for the patient.
        :param str mrn: The medical record number for the patient. An int can be provided, but will be converted and stored as a string.
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

    def insert_patient_history(self, field: str, value: float, units: str, time: int, time_units: str = None, patient_id: int = None, mrn: str = None):
        """
        Insert a patient history record using either their numeric patient id or medical record number (MRN).

        :param str field: Which part of the patients history you want to insert. Valid options are 'height' or 'weight'.
        :param float value: The value of the measurement you want to insert.
        :param str units: The units of the measurement you want to insert
        :param int time: The epoch timestamp of the time the measurement was taken.
        :param str time_units: (Optional) Units for the time. Valid options are 'ns', 's', 'ms', and 'us'. Default is nanoseconds.
        :param int patient_id: The numeric identifier for the patient.
        :param str mrn: The medical record number for the patient. An int can be provided, but will be converted and stored as a string.

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
            patient_id = self.get_patient_id(str(mrn))

        return self.sql_handler.insert_patient_history(patient_id, field, value, units, time)

    def get_patient_history_fields(self):
        """
        Returns a list of all strings in the field column of patient history.

        :return: A list of strings of all history fields
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for this method.")

        return self.sql_handler.select_unique_history_fields()

    # ------------------------------------------------------------------ #
    # Device-patient mapping and encounters
    # ------------------------------------------------------------------ #
    def get_device_patient_mapping(self, device_id_list: List[int] = None, device_tag_list: List[str] = None,
                                   patient_id_list: List[int] = None, mrn_list: List[str] = None,
                                   timestamp: int = None, start_time: int = None, end_time: int = None,
                                   time_units: str = None, truncate: bool = False):

        """
        Retrieves device-patient mappings based on the provided search criteria.

        This method allows you to obtain mappings between devices and patients, active either at a specific timestamp
        or within a given time range. It supports querying with device IDs or tags, and patient IDs or MRNs, and can
        handle single or multiple identifiers. You can also specify whether to truncate the mappings to fit within
        the provided time range.

        :param List[int] optional device_id_list: A list of device IDs.
        :param List[str] optional device_tag_list: A list of device tags.
        :param List[int] optional patient_id_list: A list of patient IDs.
        :param List[str] optional mrn_list: A list of MRNs (medical record numbers). Int values can be provided, but will be converted and stored as strings.
        :param int optional timestamp: A specific timestamp at which to find active device-patient mappings,
            in units specified by `time_units`.
        :param int optional start_time: The start time of the desired time range, in units specified by `time_units`.
        :param int optional end_time: The end time of the desired time range, in units specified by `time_units`.
        :param str optional time_units: Units for the time parameters. Valid options are `'ns'`, `'us'`, `'ms'`, and `'s'`.
            Default is `'ns'`.
        :param bool optional truncate: If `True`, the returned mappings will be truncated to fit within the specified
            time range.
        :return: A list of tuples, where each tuple contains four values in the following order:
            - device_id (int): The ID of the device associated with the patient.
            - patient_id (int): The ID of the patient associated with the device.
            - start_time (float | int): The start time of the association, in the specified time units.
            - end_time (float | int): The end time of the association, in the specified time units.
        :rtype: List[Tuple[int, int, float | int, float | int]]

        :Example:

        >>> # Retrieve mappings active at a specific timestamp
        >>> mappings = sdk.get_device_patient_mapping(
        ...     timestamp=1609459200,
        ...     device_tag_list=['device123'],
        ...     mrn_list=['123456'],
        ...     time_units='s'
        ... )
        >>> print(mappings)
        [(1, 2, 1609455600.0, 1609462800.0)]

        >>> # Retrieve mappings within a time range
        >>> mappings = sdk.get_device_patient_mapping(
        ...     start_time=1609455600,
        ...     end_time=1609462800,
        ...     device_id_list=[1, 2],
        ...     patient_id_list=[3, 4],
        ...     time_units='s',
        ...     truncate=True
        ... )
        >>> print(mappings)
        [(1, 3, 1609455600.0, 1609462800.0), (2, 4, 1609455600.0, 1609462800.0)]
        """

        time_units = "ns" if time_units is None else time_units

        if time_units not in time_unit_options.keys():
            raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")

        # Convert timestamp, start_time, and end_time to nanoseconds
        timestamp_n = int(timestamp * time_unit_options[time_units]) if timestamp is not None else None
        start_time_n = int(start_time * time_unit_options[time_units]) if start_time is not None else None
        end_time_n = int(end_time * time_unit_options[time_units]) if end_time is not None else None

        # Process device_tag_list to get device_id_list if necessary
        if device_id_list is None and device_tag_list is not None:
            device_id_list = [self.get_device_id(tag) for tag in device_tag_list]

        # Process mrn_list to get patient_id_list if necessary
        if patient_id_list is None and mrn_list is not None:
            patient_id_list = [self.get_patient_id(mrn) for mrn in mrn_list]

        # Check for API connection
        if self.metadata_connection_type == "api":
            results = self._api_get_device_patient_data(
                device_id_list=device_id_list,
                patient_id_list=patient_id_list,
                start_time=start_time_n,
                end_time=end_time_n,
                timestamp=timestamp_n,
            )
        elif timestamp_n is not None:
            results = self.sql_handler.select_device_patient_encounters(
                timestamp=timestamp_n,
                device_id_list=device_id_list,
                patient_id_list=patient_id_list,
            )
        else:
            results = self.sql_handler.select_device_patients(
                device_id_list=device_id_list,
                patient_id_list=patient_id_list,
                start_time=start_time_n,
                end_time=end_time_n,
            )

        # Truncate if necessary
        trunc_start = timestamp_n if timestamp_n is not None else start_time_n
        trunc_end = timestamp_n if timestamp_n is not None else end_time_n

        # Handle None end_time, truncate, convert time units
        mappings = []
        for device_id_result, patient_id_result, start_time_result, end_time_result in results:
            if end_time_result is None:
                end_time_result = time.time_ns()

            if truncate:
                if trunc_start is not None:
                    start_time_result = max(start_time_result, trunc_start)
                if trunc_end is not None:
                    end_time_result = min(end_time_result, trunc_end)

            if time_units != "ns":
                start_time_result = start_time_result / time_unit_options[time_units]
                end_time_result = end_time_result / time_unit_options[time_units]

            mappings.append((device_id_result, patient_id_result, start_time_result, end_time_result))

        return mappings

    def get_device_patient_data(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                                mrn_list: List[str] = None, start_time: int = None, end_time: int = None,
                                time_units: str = None):
        """
        Retrieves device-patient mappings from the dataset's database based on the provided search criteria.

        This method returns a list of tuples, where each tuple contains four values in the following order:
        - device_id (int): The ID of the device associated with the patient.
        - patient_id (int): The ID of the patient associated with the device.
        - start_time (float | int): The start time of the association between the device and the patient, in the specified time units.
        - end_time (float | int): The end time of the association between the device and the patient, in the specified time units.

        :param List[int] optional device_id_list: A list of device IDs.
        :param List[int] optional patient_id_list: A list of patient IDs.
        :param List[str] optional mrn_list: A list of MRNs (medical record numbers). Int values can be provided, but will be converted and stored as strings.
        :param int optional start_time: The start time of the device-patient association, in units specified by `time_units`.
        :param int optional end_time: The end time of the device-patient association, in units specified by `time_units`.
        :param str optional time_units: Units for the time parameters. Valid options are 'ns', 's', 'ms', and 'us'. Default is 'ns'.
        :return: A list of tuples containing device-patient mapping data.
        :rtype: List[Tuple[int, int, float | int, float | int]]

        >>> # Retrieve device-patient mappings from the dataset's database.
        >>> device_id_list = [1, 2]
        >>> patient_id_list = [3, 4]
        >>> start_time = 1647084000
        >>> end_time = 1647094800
        >>> time_units = 's'
        >>> device_patient_data = sdk.get_device_patient_data(
        ...     device_id_list=device_id_list,
        ...     patient_id_list=patient_id_list,
        ...     start_time=start_time,
        ...     end_time=end_time,
        ...     time_units=time_units
        ... )
        [(1, 3, 1647084000.0, 1647094800.0), (2, 4, 1647084000.0, 1647094800.0)]
        """
        return self.get_device_patient_mapping(
            device_id_list=device_id_list,
            patient_id_list=patient_id_list,
            mrn_list=mrn_list,
            start_time=start_time,
            end_time=end_time,
            time_units=time_units,
            truncate=True
        )

    def _api_get_device_patient_data(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                                     start_time: int = None, end_time: int = None, timestamp: int = None):
        """
        Queries the /device-patient-mapping endpoint for raw device-patient
        mappings. All time parameters are in nanoseconds.

        :param device_id_list: Device IDs to filter by.
        :param patient_id_list: Patient IDs to filter by.
        :param start_time: Range start in nanoseconds.
        :param end_time: Range end in nanoseconds.
        :param timestamp: Point-in-time query in nanoseconds (mutually exclusive with start/end).
        :return: List of (device_id, patient_id, start_time, end_time) tuples with times in nanoseconds.
        """
        params = {}

        if device_id_list is not None:
            params['device_id'] = device_id_list
        if patient_id_list is not None:
            params['patient_id'] = patient_id_list
        if timestamp is not None:
            params['timestamp'] = timestamp
        else:
            if start_time is not None:
                params['start_time'] = start_time
            if end_time is not None:
                params['end_time'] = end_time

        result = self._request("GET", "device-patient-mapping", params=params)
        return [tuple(row) for row in result]

    def get_device_patient_encounters(self, timestamp: int, device_id: int = None, device_tag: str = None,
                                      patient_id: int = None, mrn: str = None, time_units: str = None):
        """
        Retrieve device-patient encounters active at a specific time.

        This method returns a list of device-patient mappings (encounters) that were active at the given timestamp.
        You can provide either device_id or device_tag, and/or patient_id or mrn. Providing at least one of
        device or patient identifiers is required.

        :param int timestamp: The timestamp at which to find the device-patient encounters.
        :param int device_id: (Optional) The device identifier. If None, device_tag can be provided.
        :param str device_tag: (Optional) A string identifying the device. Used if device_id is None.
        :param int patient_id: (Optional) The patient identifier. If None, mrn can be provided.
        :param str mrn: (Optional) Medical record number for the patient. Used if patient_id is None. An int can be provided, but will be converted and stored as a string.
        :param str time_units: (Optional) Units for the time parameters. Valid options are 'ns', 's', 'ms', and 'us'. Default is 'ns'.

        :return: A list of tuples containing device_id, patient_id, encounter_start, and encounter_end.
        :rtype: List[Tuple[int, int, float, float]]

        >>> sdk = AtriumSDK(dataset_location="./example_dataset")
        >>> encounters = sdk.get_device_patient_encounters(
        ...     timestamp=1609459200,
        ...     device_tag="device123",
        ...     mrn=123456,
        ...     time_units="s"
        ... )
        >>> print(encounters)
        [(1, 2, 1609455600.0, 1609462800.0)]
        """
        return self.get_device_patient_mapping(
            timestamp=timestamp,
            device_id_list=[device_id] if device_id is not None else None,
            device_tag_list=[device_tag] if device_tag is not None else None,
            patient_id_list=[patient_id] if patient_id is not None else None,
            mrn_list=[mrn] if mrn is not None else None,
            time_units=time_units,
            truncate=False
        )

    def insert_encounter(self, start_time: float = None, end_time: float = None, patient_id: int = None,
                         mrn: str = None, bed_id: int = None, bed_name: str = None, source_id: int = 1,
                         visit_number: str = None, last_updated: float = None, time_units: str = 'ns'):
        """
        Inserts a new encounter into the database that represents a mapping between a patient and a bed over an interval of time.

        :param start_time: The start time of the encounter in the units specified by `time_units`.
        :param end_time: The end time of the encounter in the units specified by `time_units`, optional.
        :param patient_id: The ID of the patient.
        :param str mrn: The medical record number of the patient (mutually exclusive with `patient_id`). An int can be provided, but will be converted and stored as a string.
        :param bed_id: The ID of the bed.
        :param bed_name: The name of the bed (mutually exclusive with `bed_id`).
        :param source_id: The source ID for the encounter, default is 1.
        :param visit_number: An optional visit number for the encounter.
        :param last_updated: The timestamp of the last update in the units specified by `time_units`,
                             defaults to the current time if not provided.
        :param time_units: The units for the time parameters. Valid options are 'ns', 'us', 'ms', 's'.
                           Default is 'ns'.

        **Example:**
        >>> # Insert an encounter starting at timestamp 1609459200 seconds and ending 1 hour later
        >>> sdk.insert_encounter(start_time=1609459200, end_time=1609462800, patient_id=123, bed_name='BedA', time_units='s')
        """
        if time_units not in time_unit_options:
            raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")

        # Convert times to nanoseconds
        if start_time is None:
            raise ValueError("start_time must be provided")
        start_time_n = int(start_time * time_unit_options[time_units])
        end_time_n = int(end_time * time_unit_options[time_units]) if end_time is not None else None

        if last_updated is None:
            last_updated = time.time_ns()
        else:
            last_updated = int(last_updated * time_unit_options[time_units])

        if mrn is not None:
            patient_id = self.get_patient_id(mrn)
            if patient_id is None:
                raise ValueError(f"MRN {mrn} not found in the dataset, insert it with AtriumSDK.insert_patient")

        if patient_id is None:
            raise ValueError("patient_id or mrn must be provided")

        if bed_name is not None:
            bed_id = self.get_bed_id(bed_name)
            if bed_id is None:
                raise ValueError(f"bed_id {bed_id} not found in the dataset.")

        if bed_id is None:
            raise ValueError("bed_id or bed_name must be provided")

        self.sql_handler.insert_encounter_row(patient_id, bed_id, start_time_n, end_time_n, source_id, visit_number,
                                              last_updated)

    def get_encounters(self, timestamp: float = None, start_time: float = None, end_time: float = None,
                       bed_id: int = None, bed_name: str = None, patient_id: int = None, mrn: str = None,
                       time_units: str = 'ns'):
        """
        Queries encounters from the database based on any of the given params.

        :param timestamp: A specific timestamp in `time_units` to find all encounters that overlap the given time.
        :param start_time: The start time in `time_units` to find all encounters after (or overlapping) the given time.
        :param end_time: The end time in `time_units` to find all encounters before (or overlapping) the given time.
        :param bed_id: The ID of the bed.
        :param bed_name: The name of the bed, inplace of an id.
        :param patient_id: The ID of the patient.
        :param str mrn: The medical record number of the patient, inplace of the patient_id. An int can be provided, but will be converted and stored as a string.
        :param time_units: The units for the time parameters and returned times. Valid options: 'ns', 'us', 'ms', 's'.
                           Default is 'ns'.

        **Return Type:**
        A list of tuples representing encounters. Each tuple is of the form:
        `(id, patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated)`

        - `id` (int): The encounter ID.
        - `patient_id` (int): The ID of the patient.
        - `bed_id` (int): The ID of the bed.
        - `start_time` (float): The start time of the encounter in `time_units`.
        - `end_time` (float or None): The end time of the encounter in `time_units`, or `None` if ongoing.
        - `source_id` (int): The source ID of the encounter.
        - `visit_number` (str or None): The visit number of the encounter, if available.
        - `last_updated` (float): The last updated timestamp of the encounter in `time_units`.

        **Example:**
        >>> # Retrieve encounters active at a specific second-based timestamp
        >>> encounters = sdk.get_encounters(timestamp=1609459200, time_units='s')
        >>> print(encounters)
        [(1, 123, 10, 1609455600.0, 1609462800.0, 1, 'VISIT001', 1609459200.0)]

        >>> # Retrieve encounters within a time range (in ms), filtered by bed name
        >>> encounters = sdk.get_encounters(
        ...     start_time=1609459200000,
        ...     end_time=1609462800000,
        ...     bed_name='BedA',
        ...     time_units='ms'
        ... )
        >>> print(encounters)
        [(2, 456, 20, 1609455600000.0, 1609462800000.0, 1, None, 1609459200000.0)]
        """
        if time_units not in time_unit_options:
            raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")

        # Convert input times to nanoseconds for querying
        timestamp_n = int(timestamp * time_unit_options[time_units]) if timestamp is not None else None
        start_time_n = int(start_time * time_unit_options[time_units]) if start_time is not None else None
        end_time_n = int(end_time * time_unit_options[time_units]) if end_time is not None else None

        if mrn is not None:
            patient_id = self.get_patient_id(mrn)
            if patient_id is None:
                raise ValueError(f"MRN {mrn} not found in the dataset, insert it with AtriumSDK.insert_patient")

        if bed_name is not None:
            bed_id = self.get_bed_id(bed_name)
            if bed_id is None:
                raise ValueError(f"bed_id {bed_id} not found in the dataset.")

        results = self.sql_handler.select_encounters_from_range_or_timestamp(
            timestamp_n, start_time_n, end_time_n, bed_id, patient_id
        )

        # Convert times back from nanoseconds to the requested time_units
        converted_results = []
        for (enc_id, p_id, b_id, s_time, e_time, src_id, v_num, l_updated) in results:
            start_time_converted = s_time / time_unit_options[time_units] if s_time is not None else None
            end_time_converted = e_time / time_unit_options[time_units] if e_time is not None else None
            last_updated_converted = l_updated / time_unit_options[time_units] if l_updated is not None else None

            converted_results.append(
                (enc_id, p_id, b_id, start_time_converted, end_time_converted, src_id, v_num, last_updated_converted)
            )

        return converted_results

    def insert_device_patient_data(self, device_patient_data: List[Tuple[int, int, int, int]], time_units: str = None):
        """
        .. _insert_device_patient_data_label:

        Inserts device-patient mappings into the dataset's database.

        The `device_patient_data` parameter is a list of tuples, where each tuple contains four values in the
        following order:
        - device_id (int): The ID of the device associated with the patient.
        - patient_id (int): The ID of the patient associated with the device.
        - start_time (int | float): The start time of the association between the device and the patient, in the units specified by `time_units`.
        - end_time (int | float): The end time of the association between the device and the patient, in the units specified by `time_units`.

        The `start_time` and `end_time` values represent the time range in which the device is associated with the patient.

        :param List[Tuple[int, int, int | float, int | float]] device_patient_data: A list of tuples containing device-patient mapping
            data, where each tuple contains four values in the following order: device_id, patient_id, start_time,
            and end_time.
        :param str optional time_units: Units for the time parameters. Valid options are 'ns', 's', 'ms', and 'us'. Default is 'ns'.
        :return: None

        >>> # Insert a device-patient mapping into the dataset's database.
        >>> device_patient_data = [(1, 2, 1647084000, 1647094800),
        ...                        (1, 3, 1647084000, 1647094800)]
        >>> time_units = 's'
        >>> sdk.insert_device_patient_data(device_patient_data, time_units=time_units)
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for insertion.")

        time_units = "ns" if time_units is None else time_units

        if time_units not in time_unit_options.keys():
            raise ValueError(f"Invalid time units. Expected one of: {', '.join(time_unit_options.keys())}")

        # Convert start_time and end_time to nanoseconds
        converted_device_patient_data = []
        for device_id, patient_id, start_time, end_time in device_patient_data:
            start_time_ns = int(start_time * time_unit_options[time_units])
            end_time_ns = None if end_time is None else int(end_time * time_unit_options[time_units])
            converted_device_patient_data.append((int(device_id), int(patient_id), start_time_ns, end_time_ns))

        self.sql_handler.insert_device_patients(converted_device_patient_data)

    def convert_patient_to_device_id(self, start_time: int, end_time: int = None, patient_id: int = None,
                                     mrn: str = None):
        """
        Converts a patient ID or MRN to a device ID based on the specified time range.

        :param int start_time: Start time or only time for the association.
        :param int end_time: End time for the association. If None, then start_time is taken as a single point in time.
        :param int patient_id: Patient ID to be converted.
        :param str mrn: MRN to be converted. An int can be provided, but will be converted and stored as a string.
        :return: Device ID if a single device fully encapsulates the time range, otherwise None.
        :rtype: int or None
        """

        end_time = start_time if end_time is None else end_time
        # Retrieve device-patient mapping data
        if patient_id is not None:
            device_patient_data = self.get_device_patient_mapping(patient_id_list=[patient_id], start_time=start_time,
                                                                  end_time=end_time, truncate=False)
        elif mrn is not None:
            device_patient_data = self.get_device_patient_mapping(mrn_list=[mrn], start_time=start_time,
                                                                  end_time=end_time, truncate=False)
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

    def convert_device_to_patient_id(self, start_time: int, end_time: int = None, device=None,
                                     conflict_resolution='error'):
        """
        Converts a device ID or tag to a patient ID based on the specified time range.

        :param int start_time: Start time or only time for the association.
        :param int end_time: End time for the association. If None, then start_time is taken as a single point in time.
        :param device: Device ID (int) or tag (str) to be converted.
        :param str conflict_resolution: How to handle multiple matching patients. Options are 'error', '90_percent_overlap', 'always_none'.
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

        end_time = start_time if end_time is None else end_time

        # Retrieve device-patient mapping data
        device_patient_data = self.get_device_patient_mapping(device_id_list=[device_id], start_time=start_time,
                                                              end_time=end_time, truncate=False)

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

        # Handle multiple matching patients based on conflict_resolution parameter
        if len(matching_patients) > 1:
            if conflict_resolution == 'error':
                raise ValueError(
                    f"Multiple patients ({matching_patients}) found for the same time range with parameters: "
                    f"start_time={start_time}, end_time={end_time}, device={device}. "
                    "Please check and fix the device_patient table.")
            elif conflict_resolution == '90_percent_overlap':
                for patient_id in matching_patients:
                    total_overlap = sum(min(end_time, interval[1]) - max(start_time, interval[0])
                                        for interval in patient_intervals[patient_id])
                    if total_overlap >= 0.9 * (end_time - start_time):
                        print(f"Warning: Patient {patient_id} overlaps 90% or more of the time range.")
                        return patient_id
                print("Warning: No patient overlaps 90% or more of the time range.")
                return None
            elif conflict_resolution == 'always_none':
                print("Warning: Multiple patients found. Returning None.")
                return None
            else:
                raise ValueError(f"Invalid conflict_resolution value: {conflict_resolution}")

        return matching_patients[0] if matching_patients else None

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    def get_labels(self, label_name_id_list: List[int] = None, name_list: List[str] = None, device_list: List[Union[int, str]] = None,
                   start_time: int = None, end_time: int = None, time_units: str = None, patient_id_list: List[int] = None,
                   label_source_list: List[Union[str, int]] = None, include_descendants=True, limit: int = None, offset: int = 0,
                   measure_list: List[Union[int, tuple[str, int | float, str]] | None] = None):
        """
        Retrieve labels from the database based on specified criteria.

        :param List[int] label_name_id_list: List of label set IDs to filter by.
        :param List[str] name_list: List of label names to filter by. Mutually exclusive with `label_name_id_list`.
        :param List[Union[int, str]] device_list: List of device IDs or device tags to filter by.
        :param int start_time: Start time filter for the labels.
        :param int end_time: End time filter for the labels.
        :param str time_units: Units for the `start_time` and `end_time` filters. Valid options are 'ns', 's', 'ms', and 'us'.
        :param List[int] patient_id_list: List of patient IDs to filter by.
        :param List[Union[str, int]] label_source_list: List of label source names or IDs to filter by.
        :param bool include_descendants: Returns all labels of descendant label_name, using requested_name_id and
            requested_name to represent the label name of the requested parent.
        :param int limit: Maximum number of rows to return.
        :param int offset: Offset this number of rows before starting to return labels. Used in combination with limit.
        :param int measure_list: The list of measure_ids or measure tuples (measure_tag, freq_hz, measure_units) you
            would like to restrict the search to. If you specify measures but also want all the labels that don't have a
            specified measure_id (the labels for all signals at that time) add None to the list. Measures can also be
            None to get all labels for a specific source regardless of measure_id.

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
                    'measure_id': 2
                },
                ...
            ]

        Note:
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
            return self._api_get_labels(label_name_id_list, name_list, device_list, start_time, end_time, patient_id_list,
                                        label_source_list, include_descendants, limit, offset, measure_list)

        # Convert label names to IDs if name_list is used
        if name_list:
            name_id_list = [self.get_label_name_id(name) for name in name_list]
            for label_name, label_id in zip(name_list, name_id_list):
                if label_id is None:
                    raise ValueError(f"Label name '{label_name}' not found in the database.")
            label_name_id_list = name_id_list


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

        # Convert measure tags to IDs
        if measure_list:
            measure_id_list = []
            for measure in measure_list:
                measure_id = self.get_measure_id(measure[0], measure[1], measure[2], freq_units='Hz') if isinstance(measure, tuple) else measure
                if measure_id is None:
                    raise ValueError(f"Measure Tag {measure} not found in database")
                measure_id_list.append(measure_id)
            measure_list = measure_id_list

        if (not measure_list and not label_source_id_list and not patient_id_list and device_list
                and label_name_id_list and all(dev_id in self.label_cache for dev_id in device_list)):
            return self._get_cached_labels(label_name_id_list=label_name_id_list, device_list=device_list,
                                           start_time=start_time, end_time=end_time, include_descendants=include_descendants)

        closest_requested_ancestor_dict = {}
        if label_name_id_list and include_descendants:
            label_name_id_list, closest_requested_ancestor_dict = collect_all_descendant_ids(
                label_name_id_list, self.sql_handler)

        labels = self.sql_handler.select_labels_with_info(
            label_set_id_list=label_name_id_list,
            device_id_list=device_list,
            patient_id_list=patient_id_list,
            start_time_n=start_time,
            end_time_n=end_time,
            label_source_id_list=label_source_id_list if label_source_id_list else None,
            measure_id_list=measure_list,
            limit=limit, offset=offset,
        )

        # Extract unique label_set_ids and device_ids
        unique_label_set_ids = {label[2] for label in labels}
        unique_device_ids = {label[3] for label in labels}

        # Create dictionaries for label set and device info for optimized lookup
        label_set_id_to_info = {label_set_id: self.get_label_name_info(label_set_id) for label_set_id in
                                unique_label_set_ids}
        device_id_to_info = {device_id: self.get_device_info(device_id) for device_id in unique_device_ids}

        result = []
        for (label_entry_id, label_name, label_set_id, device_id, measure_id, label_source_name, label_source_id,
             start_time_n, end_time_n, patient_id) in labels:

            requested_id = closest_requested_ancestor_dict.get(label_set_id, label_set_id)
            requested_name = self.get_label_name_info(requested_id)['name']

            # patient_id = self.convert_device_to_patient_id(
            #     start_time=start_time_n, end_time=end_time_n, device=device_id,
            #     conflict_resolution='90_percent_overlap')
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
                'label_source': label_source_name,
                'measure_id': measure_id
            }
            result.append(formatted_label)

        return result

    def _api_get_labels(self, label_name_id_list=None, name_list=None, device_list=None, start_time=None, end_time=None,
                        patient_id_list=None, label_source_list: Optional[List[Union[str, int]]] = None,
                        include_descendants=True, limit=None, offset=0, measure_list: List[Union[int, tuple[str, int | float, str], None]] = None):
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
                'measure_list': measure_list,
                'limit': limit, 'offset': offset,
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

    def _get_cached_labels(self, label_name_id_list: List[int] = None, device_list: List[int] = None,
                           start_time: int = None, end_time: int = None, include_descendants: bool = True,
                           limit: int = None, offset: int = 0):

        all_labels = []

        for device_id in device_list:
            if device_id not in self.label_cache:
                continue

            for label_name_id in label_name_id_list:
                matching_labels = self._find_labels(
                    label_name_id=label_name_id,
                    device_id=device_id,
                    start_time=start_time if start_time is not None else 0,
                    end_time=end_time if end_time is not None else float('inf'),
                    include_descendants=include_descendants
                )
                all_labels.extend(matching_labels)

        # Remove duplicates and sort
        seen = set()
        unique_labels = []
        for label in all_labels:
            label_id = label[0]  # label_entry_id
            if label_id not in seen:
                seen.add(label_id)
                unique_labels.append(label)

        # Sort by start time, then end time, then label ID
        unique_labels.sort(key=lambda x: (x[7], x[8], x[0]))

        # Apply offset and limit
        if offset > 0:
            unique_labels = unique_labels[offset:]
        if limit is not None:
            unique_labels = unique_labels[:limit]

        # Format the results similar to get_labels
        result = []
        for label_record in unique_labels:
            (label_entry_id, label_name, label_set_id, device_id, measure_id,
             label_source_name, label_source_id, start_time_n, end_time_n, patient_id) = label_record

            # Get the requested ancestor info if using descendants
            requested_id = label_set_id
            requested_name = label_name

            if include_descendants:
                # Find which original label_name_id this descendant belongs to
                for orig_id in label_name_id_list:
                    if orig_id in self.descendant_cache:
                        _, ancestor_dict = self.descendant_cache[orig_id]
                        if label_set_id in ancestor_dict:
                            requested_id = ancestor_dict[label_set_id]
                            requested_name = self.label_lookup_caches['label_set_id_to_info'][requested_id]['name']
                            break

            mrn = None if patient_id is None else self.get_mrn(patient_id)
            device_info = self.label_lookup_caches['device_id_to_info'][device_id]

            formatted_label = {
                'label_entry_id': label_entry_id,
                'label_name_id': label_set_id,
                'label_name': label_name,
                'requested_name_id': requested_id,
                'requested_name': requested_name,
                'device_id': device_id,
                'device_tag': device_info['tag'],
                'patient_id': patient_id,
                'mrn': mrn,
                'start_time_n': start_time_n,
                'end_time_n': end_time_n,
                'label_source_id': label_source_id,
                'label_source': label_source_name,
                'measure_id': measure_id
            }
            result.append(formatted_label)

        return result

    def insert_label(self, name: str, start_time: int, end_time: int, device: Union[int, str] = None,
                     patient_id: int = None, mrn: str = None, time_units: str = None,
                     label_source: Union[str, int] = None, measure: Union[int, tuple[str, int | float, str]] = None):
        """
        Insert a label record into the database.

        :param str name: Name of the label type.
        :param int start_time: Start time for the label.
        :param int end_time: End time for the label.
        :param Union[int, str] device: Device ID or device tag (exclusive with device and patient_id).
        :param int patient_id: Patient ID for the label to be inserted (exclusive with device and mrn).
        :param str mrn: MRN for the label to be inserted (exclusive with device and patient_id). An int can be provided, but will be converted and stored as a string.
        :param str time_units: Units for the `start_time` and `end_time`. Valid options are 'ns', 's', 'ms', and 'us'.
        :param Union[str, int] label_source: Name or ID of the label source.
        :param Union[int, tuple[str, int|float, str]] measure: Either the measure ID or the measure tuple
            (measure_tag, freq_hz, measure_units), if the label is for a specific measure. Leave as none if it's for all measures.
        :raises ValueError: If the provided label_source is not found in the database.
        :return: The ID of the inserted label

        Example usage:

        .. code-block:: python

            # Insert a label for a device with ID 42
            insert_label(name='Sleep Stage', start_time=1609459200_000_000_000, end_time=1609462800_000_000_000, device=42, measure=20)

            # Insert a label for a patient with patient_id 12345
            insert_label(name='Medication', start_time=1609459200_000, end_time=1609462800_000, patient_id=12345, time_units='ms', measure=None)

            # Using a device tag instead of device ID
            insert_label(name='Arrhythmia', start_time=1609459200_000_000_000, end_time=1609462800_000_000_000, device='device-tag-xyz', measure=('ECG', 200, 'Milli_Volts'))

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
        elif isinstance(device, int) or isinstance(device, np.generic):
            converted_device_id = int(device)

        if converted_device_id is None or (isinstance(device, int) and self.get_device_info(device) is None):
            raise ValueError(f"device not found for device {device} patient_id {patient_id} mrn {mrn}")

        # convert measure tag tuple into a measure ID if necessary
        if isinstance(measure, tuple):
            measure_id = self.get_measure_id(measure[0], measure[1], measure[2], freq_units='Hz')
            if measure_id is None:
                raise ValueError(f"Measure Tag {measure} not found in database")
            measure = measure_id

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
        return self.sql_handler.insert_label(label_id, converted_device_id, start_time, end_time, label_source_id, measure)

    def insert_labels(self, labels: List[Tuple[str, Union[int, str], Union[int, tuple[str, int | float, str], None], Union[str, int, None], int, int]],
                      time_units: str = None, source_type: str = None):
        """
        Insert multiple label records into the database.

        :param List[Tuple[str, Union[int, str], Union[int, tuple[str, int | float, str]], int, int, Union[str, int]]] labels: A list of labels. Each label is a tuple containing:
            - Name of the label type.
            - Source ID based on the source_type parameter (device ID, device tag, patient ID, or MRN).
            - Measure for the label. Can be measure ID, tuple containing (measure_tag, freq_hz, measure_units) or none if it applies to all measures at that time.
            - Name or ID of the label source. (Can be None, for no specified source)
            - Start time for the label.
            - End time for the label.

        :param str time_units: Units for the `start_time` and `end_time` of each label. Valid options are 'ns', 's', 'ms', and 'us'. (default ns)
        :param str source_type: The type of source ID provided in the labels. Valid options are 'device_id', 'device_tag', 'patient_id', and 'mrn'.
        :raises ValueError: If the provided label_source, source_type or measure is not found in the database.

        Example usage:

        .. code-block:: python

            # Using device ID as the source type and measure ID
            labels_data = [
                ('Sleep Stage', 42, 3, None, 1609459200_000_000_000, 1609462800_000_000_000),
                ('Medication', 56, None, 'Medication DB', 1609459200_000_000_000, 1609462800_000_000_000)
            ]
            insert_labels(labels=labels_data, time_units='s', source_type='device_id')

            # Using MRN as the source type and measure tuple
            labels_data = [
                ('Heart Rate', 1234567, ('ECG', 200, 'Milli_Volts'), None, 1609459200_000, 1609462800_000),
                ('Blood Pressure', 1234568, ('ECG', 200, 'Milli_Volts'), 'Automatic Device', 1609459200_000, 1609462800_000)
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
            name, source_id, measure, label_source, start_time, end_time = label

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
                device = int(source_id)
                if self.get_device_info(source_id) is None:
                    raise ValueError(f"device id {source_id} not found in database")
            else:
                raise ValueError(f"Invalid source type. Expected one of: {', '.join(valid_source_types)}")

            # convert measure tag tuple into a measure ID if necessary
            if isinstance(measure, tuple):
                measure_id = self.get_measure_id(measure[0], measure[1], measure[2], freq_units='Hz')
                if measure_id is None:
                    raise ValueError(f"Measure Tag {measure} not found in database")
                measure = measure_id

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
            formatted_labels.append((label_id, device, measure, label_source_id, start_time, end_time))

        # Insert the labels into the database
        self.sql_handler.insert_labels(formatted_labels)

    def delete_labels(self, label_id_list: List[int] = None, label_name_id_list: List[int] = None, name_list: List[str] = None,
                      device_list: List[Union[int, str]] = None, start_time: int = None, end_time: int = None, time_units: str = None,
                      patient_id_list: List[int] = None, label_source_list: Optional[List[Union[str, int]]] = None,
                      measure_list: List[Union[int, tuple[str, int | float, str], None]] = None):

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
        :param int measure_list: The list of measure_ids you would like to delete. Can also be a list of tuples
            specifying the measure (measure_tag, freq_hz, measure_units). If None it will delete labels regardless of measure_id.
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

        if all(param is None for param in
               [label_id_list, label_name_id_list, name_list, device_list, start_time, end_time, patient_id_list,
                label_source_list, measure_list]):
            raise ValueError("No parameters were provided. For safety, you need to specify at least one parameter. Use label_id_list='*' to delete all labels.")

        if label_id_list == "*":
            all_labels = self.get_labels()
            all_label_ids = [label_info['label_entry_id'] for label_info in all_labels]
            return self.sql_handler.delete_labels(all_label_ids)

        elif label_id_list is not None:
            return self.sql_handler.delete_labels(label_id_list)

        filtered_labels = self.get_labels(label_name_id_list=label_name_id_list, name_list=name_list,
                                          device_list=device_list, start_time=start_time, end_time=end_time,
                                          time_units=time_units, patient_id_list=patient_id_list,
                                          label_source_list=label_source_list, measure_list=measure_list)
        filtered_label_ids = [label_info['label_entry_id'] for label_info in filtered_labels]
        return self.sql_handler.delete_labels(filtered_label_ids)

    # ------------------------------------------------------------------ #
    # Label names, label sources and label time series
    # ------------------------------------------------------------------ #
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

        result = self.sql_handler.select_label_name_parent(label_name_id)
        if result:
            return self.get_label_name_info(result[0])
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

        # Check the cache first
        if name in self._label_source_ids:
            return self._label_source_ids[name]

        # If not in cache, fetch from the database or API
        if self.metadata_connection_type == 'api':
            params = {'label_source_id': None, 'label_source_name': name}
            source_id = self._request("GET", "/labels/source", params=params)
        else:
            source_id = self.sql_handler.select_label_source_id_by_name(name)

        if source_id is not None:
            # Update the cache
            self._label_source_ids[name] = source_id
            self._label_sources[source_id] = {'id': source_id, 'name': name}
        return source_id

    def get_label_source_info(self, label_source_id: int) -> Optional[dict]:
        """
        Retrieve information about a specific label source by its ID.
        :param label_source_id: The identifier for the label source.
        :return: A dictionary containing information about the label source, or None if not found.
        """

        # Check the cache first
        if label_source_id in self._label_sources:
            return self._label_sources[label_source_id]

        # If not in cache, fetch from the database or API
        if self.metadata_connection_type == 'api':
            params = {'label_source_id': label_source_id, 'label_source_name': None}
            source_info = self._request("GET", "/labels/source", params=params)
        else:
            source_info = self.sql_handler.select_label_source_info_by_id(label_source_id)

        if source_info is not None:
            # Update the cache
            self._label_sources[label_source_id] = source_info
            self._label_source_ids[source_info['name']] = label_source_id
        return source_info

    def get_all_label_sources(self, limit=None, offset=0) -> dict:
        """
        Retrieve all distinct label sources from the database.
        :param int limit: Maximum number of rows to return.
        :param int offset: Offset this number of rows before starting to return label sources.
        :return: A dictionary where keys are label source IDs and values are dictionaries containing 'id' and 'name' keys.
        :rtype: dict
        """
        if self.metadata_connection_type == "api":
            warnings.warn("API mode cannot cache label_sources, leaving cache empty.")
            return {}

        source_tuple_list = self.sql_handler.select_all_label_sources(limit=limit, offset=offset)

        source_dict = {}
        for source_info in source_tuple_list:
            source_id, source_name = source_info
            source_dict[source_id] = {
                'id': source_id,
                'name': source_name
            }
        return source_dict

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
                              out: np.ndarray = None, label_source_list: Optional[List[Union[str, int]]] = None,
                              measure: Union[int, tuple[str, int | float, str]] = None,
                              include_descendants: bool = True):
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
        :param int measure: The measure_id or tuple specifying the measure (measure_tag, freq_hz, measure_units), you
            would like to restrict the search to. If none it will get all labels regardless of measure_id.
        :param np.ndarray out: An optional pre-allocated numpy array to hold the result. The shape must match the expected result shape,
            which is the same as `timestamp_array`. Allowed dtypes are integer types or boolean. If provided,
            the results are written into this array in-place. It should be initialized with zeros.
            Otherwise, a new array is allocated.
        :param Optional[List[Union[str, int]]] label_source_list: List of label source names or IDs to filter by.
        :param bool include_descendants: Whether to include descendant labels when querying the database.

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
                                 label_source_list=label_source_list, measure_list=[measure] if measure is not None else None,
                                 include_descendants=include_descendants)

        # Create a binary array to indicate presence of a label for each timestamp, if not provided.
        if out is not None:
            if out.shape != timestamp_array.shape:
                raise ValueError(
                    f"The 'out' array shape {out.shape} doesn't match expected shape {timestamp_array.shape}.")

            if out.dtype.kind not in ('b', 'i', 'u'):  # boolean, signed int, unsigned int
                raise ValueError(f"The 'out' array dtype is {out.dtype}, but expected boolean or integer type.")

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

    # ------------------------------------------------------------------ #
    # Windowed iteration (get_iterator)
    # ------------------------------------------------------------------ #
    def _preflight_window_sizing(self, definition, window_duration, window_slide,
                                 period_overrides, time_units):
        """Validate the window geometry against every measure and size the batch.

        Two jobs that both need the same per-measure "how many values does a
        window/slide actually contain" number, which is why they live together:

        1. reject a window duration or slide that would contain zero values for
           some measure -- a condition that otherwise surfaces far downstream as
           an opaque "slice step cannot be zero" from ``sliding_window_view``;
        2. return the values-per-slide of the FASTEST measure, which is what the
           default ``num_windows_prefetch`` / ``cached_windows_per_source``
           divide by.

        Waveform measures are checked with the exact original frequency
        arithmetic and error text; aperiodic kinds are checked against their
        resolved nominal raster period instead, because ``freq_nhz`` is not a
        raster rate for them.

        :returns: values per window slide for the fastest measure (at least 1).
        """
        # Pre-flight sample-count guard.
        #
        # ``measure_info['freq_nhz']`` is not a raster rate for an aperiodic
        # measure, so deriving the per-window sample count from it alone would
        # reject a genuinely 1/300 Hz NIBP outright -- and ``period_overrides``
        # could not rescue it, because this runs before the iterator exists.
        # Aperiodic kinds are rasterized onto a nominal grid period
        # (period_overrides -> 1 s default), so use that resolved period for them.
        # Waveform measures use the exact frequency arithmetic below.
        _period_overrides = period_overrides or {}
        _effective_freqs_nhz = []
        for measure_info in definition.validated_data_dict['measures']:
            signal_kind = measure_info.get('signal_kind') or DEFAULT_SIGNAL_KIND
            if signal_kind == SIGNAL_KIND_WAVEFORM:
                # Legacy waveform arithmetic and error text. (A period override on
                # a waveform is rejected with a measure-named error when the
                # iterator is built.)
                freq_nhz = int(measure_info['freq_nhz'])
                if (int(window_duration) * freq_nhz) // (10 ** 18) == 0:
                    raise ValueError(
                        f"Window Slide {window_duration} with units {time_units} is less than a single "
                        f"value. Please increase it to at least one sample period.")
                if (int(window_slide) * freq_nhz) // (10 ** 18) == 0:
                    raise ValueError(
                        f"Window Slide {window_slide} with units {time_units} is less than a single "
                        f"value. Please increase it to at least one sample period.")
                _effective_freqs_nhz.append(freq_nhz)
                continue

            # Aperiodic kinds are rasterized onto a nominal grid period, so check
            # THAT, with the same measure-named message the iterator uses.
            nominal_period_ns = int(resolve_nominal_period_ns(
                measure_info, period_override=_period_overrides.get(measure_info['id'])))
            if nominal_period_ns > int(window_duration):
                raise ValueError(
                    f"Measure {measure_info['id']}: resolved nominal raster period "
                    f"{nominal_period_ns} ns is larger than the window duration "
                    f"{int(window_duration)} ns, so a window would contain zero grid cells. "
                    f"Increase window_duration or lower this measure's period via period_overrides.")
            if nominal_period_ns > int(window_slide):
                raise ValueError(
                    f"Measure {measure_info['id']}: resolved nominal raster period "
                    f"{nominal_period_ns} ns is larger than the window slide "
                    f"{int(window_slide)} ns, so the slide would advance zero grid cells. "
                    f"Increase window_slide or lower this measure's period via period_overrides.")
            _effective_freqs_nhz.append(max(1, (10 ** 18) // max(1, nominal_period_ns)))

        # Batch sizing below divides by this, so floor it at 1: the per-measure
        # guards above ensure at least one grid cell per slide, but the
        # freq-from-period round trip can still floor a legitimate aperiodic
        # period (e.g. 3 s) to 0 values.
        #
        # Divide by the FASTEST measure's values-per-slide, not the slowest. The
        # batch's memory footprint is driven by the fastest measure: the iterator
        # allocates `max_batch_size * row_size` float64s, and `row_size` is derived
        # from the LOWEST period in the definition (dataset_iterator.py: `row_size =
        # window_duration_ns // lowest_period_ns`). Dividing by the slowest measure
        # instead would inflate the default batch by the definition's rate ratio --
        # adding one 1 Hz measure to a 250 Hz, 10 s-window definition would take the
        # default batch from ~10 MB to ~2621 MB (shuffling: ~104 MB to ~26 GB).
        # For a single-rate definition min == max, so this is a no-op there.
        max_freq_nhz = max(_effective_freqs_nhz)
        number_of_values_per_window_slide = max(
            1, (int(window_slide) * int(max_freq_nhz)) // (10 ** 18))

        return number_of_values_per_window_slide

    def get_iterator(self, definition, window_duration, window_slide, gap_tolerance=None, num_windows_prefetch=None,
                     time_units: str = None, label_threshold=0.5, iterator_type=None, window_filter_fn=None,
                     shuffle=False, cached_windows_per_source=None, patient_history_fields=None, start_time=None,
                     end_time=None, num_iterators=1, label_exact_match=False,
                     aperiodic_fill=None, fill_overrides=None, period_overrides=None) -> Union[
        DatasetIterator, List[DatasetIterator]]:
        """
        Constructs and returns a `DatasetIterator` object or a list of `DatasetIterator` objects that allow iteration
        over the dataset according to the specified definition.

        The method first verifies the provided definition against the dataset of the calling class object.
        If certain parts of the cohort definition aren't present within the dataset, the method will truncate the
        requested cohort to fit the dataset and issue warnings about the dropped data.

        When using a Pytorch DataLoader, ensure that `get_iterator`'s `num_windows_prefetch` is greater than the DataLoader
        `batch_size` * `num_workers` * `prefetch_factor`. If you do this, then the Dataloader will correctly cooperate
        with the Iterator's cache functionality.

        For large datasets, it is recommended to run `AtriumSDK.load_device(device_id)` for all devices requested in the definition.
        This will cache the file locations of all waveform data in RAM which significantly reduces the overhead of each
        `AtriumSDK.get_data` call internally performed by the iterator.

        - **Caching and Shuffling Logic**: When shuffling, the caching system is designed to balance randomness and efficiency.

          The parameter `num_windows_prefetch` controls the total number of windows fetched and cached each time a window is
          requested outside the current cache, while `cached_windows_per_source` specifies
          the minimum number of windows retrieved from each source (typically patients or devices).

          For example, if you set `num_windows_prefetch=1000` and `cached_windows_per_source=100`, the iterator will randomly select 10 sources
          (`1000 / 100 = 10`) and retrieve 100 windows from each. Once all 1000 windows are iterated over, another set of 10 random sources will be
          selected, and 100 windows will be fetched from each. This source selection is randomized, and the seed for randomness can be controlled
          by the `shuffle` parameter. If `shuffle=True`, the seed is random. If `shuffle` is set to an integer, that integer will be used as the seed for reproducibility.

          If there are fewer sources than needed to fill the `num_windows_prefetch` value, the system will adjust accordingly. For instance, if
          `num_windows_prefetch=1000` but only 5 sources are available, the system will retrieve 200 windows per source (`1000 / 5 = 200`),
          even though `cached_windows_per_source=100`. This means `cached_windows_per_source` acts as the **minimum** number of windows fetched per source,
          but more can be retrieved if necessary to meet the prefetch requirement.

          An efficient strategy is to set `cached_windows_per_source` to cover a single block of data (e.g., the size of a data block in `AtriumSDK.block.block_size`).
          This will ensure that each read from the dataset is efficiently used (very little data will be discarded)
          Then, to increase randomness, `num_windows_prefetch` should be a large multiple of `cached_windows_per_source`
          to ensure that the cache includes windows from many different. For instance, a common approach would be to
          set `num_windows_prefetch` at least 100 times larger than `cached_windows_per_source`, ensuring that the
          cache spans 100 randomly chosen sections of the dataset.

          Alternatively, for the highest level of randomness, you can set `cached_windows_per_source` to 1. This means each window in the cache will be independently
          chosen from every other window. This strategy will yield very poor performance because the iterator must
          perform a single read per window and discard all read data not within the bounds of the window.

          Regardless of the above parameters if shuffle is True or an int, all windows in the cache will be randomly
          shuffled before being passed to the user.

        - **Aperiodic and string measures**: ``waveform`` measures use the usual NaN-filled
          sample grid. Aperiodic measures (``sample``/``event``/``state``) are rasterized onto the window
          grid using a per-``signal_kind`` fill rule: ``sample`` -> ``carry_forward`` (default; also
          ``sparse``, ``aggregate:last|mean|min|max``), ``state`` -> ``carry_forward`` with left-censoring,
          ``event`` -> ``presence`` (default; also ``count``). Aperiodic measures rasterize at a nominal
          period (``period_overrides`` -> the measure's stored period for waveform -> a 1 second default
          for aperiodic kinds). Configure fill via ``aperiodic_fill`` (global default),
          ``fill_overrides`` (``{measure_id: rule}``) and ``period_overrides`` (``{measure_id: period}``).

          Unknown / left-censored cells (data gaps, empty ``sparse`` cells, the region of a ``state`` before
          its first observed transition, and any cell of a trailing partial window that lies **outside the
          definition range**) are marked **with a sentinel in the ``values`` array**: ``NaN`` for
          numeric channels and ``-1`` (``UNKNOWN_STRING_CODE``, decodes to ``"<unknown>"``) for string/code
          channels. ``presence``/``count`` cells INSIDE the range are a meaningful ``0`` ("no event
          occurred"); only their out-of-range tail is ``NaN``, so ``actual_count`` reports true coverage.
          **Limitation:** the sentinel conflates "unknown/censored" with a genuine missing reading
          -- there is no separate ``known`` mask.

          String measures carry int64 dictionary codes in the window; decode on demand with
          ``Window.decode_string_signal(sdk, measure_key)`` or
          ``iterator.decode_window_strings(window, measure_key)``. Those accessors decode **code channels
          only**: a string ``event`` measure rendered as ``presence``/``count`` holds occupancy floats, not
          codes, and decoding it would fabricate vocabulary strings, so it raises. Read the underlying
          strings with :meth:`get_string_data` / :meth:`get_event_intervals` instead.

          This fill configuration is applied by the default/mapped iterators. ``DatasetDefinition.filter``
          uses the same default per-kind configuration (without these iterator-specific override
          parameters), and the ``lightmapped`` iterator -- numeric grid only -- **rejects** a definition
          containing a string or aperiodic measure with a measure-named error pointing at
          ``iterator_type='mapped'``. The
          rasterizer does not right-censor a ``state`` after its last observation, and does not pair
          ``from``/``to`` events into intervals -- use :meth:`get_event_intervals` for the latter.

        :param definition: A DatasetDefinition object or string representation specifying the measures and
                           patients or devices over particular time intervals.
        :param int window_duration: Duration of each window in units time_units (default nanoseconds).
        :param int window_slide: Slide duration between consecutive windows in units time_units (default nanoseconds).
        :param int gap_tolerance: Tolerance for gaps in definition intervals auto generated by "all", if not already validated
            (optional) in units time_units (default nanoseconds). The default gap_tolerance is 0.
        :param int num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
        :param str time_units: If you would like the window_duration, window_slide and gap_tolerance to be specified in units other than
                            nanoseconds you can choose from one of ["s", "ms", "us", "ns"].
        :param float label_threshold: The percentage of the window that must contain a label before the entire window is
            marked by that label (eg. 0.5 = 50%). All labels meeting the threshold will be marked.
        :param str iterator_type: Specify the type of iterator. If set to 'mapped', a RandomAccessDatasetIterator
          will be returned, allowing indexed access to dataset windows. If set to `lightmapped` a lightweight low RAM mapped iterator is returned.
          'lightmapped' is most suitable when you want true random shuffles and/or you're going to be jumping around
          the indices in no particular order.
          By default or if set to None, a standard DatasetIterator is returned.
        :param bool | int shuffle: If True, the order of windows will be randomized before iteration. If set to an integer, this
            value will seed the random number generator for reproducible shuffling. If False, windows are
            returned in their original order.
        :param int cached_windows_per_source: The maximum number of windows to cache for a single source before moving
            on to a new source, helpful for adding more randomness to the shuffle. Making it too small heavily decreases
            efficiency, making it too large will make the windows less random when shuffled. Only used when shuffling.
        :param list patient_history_fields: A list of patient_info fields you would like returned in the Window object.
        :param int start_time: The global minimum start time for data windows, using time_units units.
        :param int end_time: The global maximum end time for data windows, using time_units units.
        :param str aperiodic_fill: Default fill rule for aperiodic measures.
            One of ``"carry_forward"``, ``"sparse"``, ``"aggregate:last|mean|min|max"`` (for
            ``sample`` kinds) or ``"presence"`` / ``"count"`` (for ``event`` kinds). When ``None`` each
            measure uses its per-``signal_kind`` default (``sample``/``state`` -> carry-forward,
            ``event`` -> presence). A rule name that is not a supported fill rule at all (e.g. the
            hyphen typo ``"carry-forward"``) raises; a *valid* rule that is merely incompatible with a
            given measure's kind silently falls back to that kind's default. ``waveform`` numeric
            measures are unaffected (unchanged NaN grid). Unknown / left-censored cells carry a
            sentinel: ``NaN`` for numeric channels, a reserved unknown code for string channels (a
            sentinel conflates unknown/censored with a genuine missing reading).
        :param dict fill_overrides: ``{measure_id: rule}`` per-measure fill rule overrides. Unlike
            ``aperiodic_fill``, an override incompatible with the measure's kind raises. Keys must be
            measure IDs present in the definition -- a key matching no measure (for instance a measure
            *tag*) raises rather than being silently ignored.
        :param dict period_overrides: ``{measure_id: period}`` per-measure nominal raster period
            overrides in ``time_units``. Highest precedence in period resolution
            (override -> 1 s default for aperiodic kinds). **Aperiodic kinds only**: a waveform is
            always sampled on its own stored period, so an override on a ``waveform`` measure raises.
            Unknown keys raise, as with ``fill_overrides``.

        :returns: A single DatasetIterator, or a list of DatasetIterator objects, depending on the
            value of num_iterators.
        :rtype: Union[DatasetIterator, List[DatasetIterator]]

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
            definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)

            # Get the Iterator Object
            slide_size_nano = window_size_nano = 60_000_000_000  # 1 minute nano
            iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano)

            # Loop over all windows (Window objects)
            for window in iterator:
                print(window)

        """
        if iterator_type is None:
            iterator_type = "iterator"
        # check that a correct unit type was entered
        time_units = "ns" if time_units is None else time_units
        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        # convert to nanoseconds
        window_duration = int(window_duration * time_unit_options[time_units])
        window_slide = int(window_slide * time_unit_options[time_units])
        if gap_tolerance is not None:
            gap_tolerance = int(gap_tolerance * time_unit_options[time_units])

        start_time_n, end_time_n = start_time, end_time
        if start_time_n is not None:
            start_time_n = int(start_time_n * time_unit_options[time_units])

        if end_time_n is not None:
            end_time_n = int(end_time_n * time_unit_options[time_units])

        # Convert per-measure nominal period overrides from time_units to ns.
        if period_overrides is not None:
            period_overrides = {int(mid): int(period * time_unit_options[time_units])
                                for mid, period in period_overrides.items()}

        max_cache_duration_per_source = None
        if cached_windows_per_source is not None:
            assert isinstance(cached_windows_per_source, int), "cached_windows_per_source must be of type int."
            assert cached_windows_per_source > 0, "cached_windows_per_source must be at least 1."
            max_cache_duration_per_source = window_duration + (window_slide * (cached_windows_per_source - 1))

        # Check if we need to partition the dataset
        if num_iterators > 1:
            random_state = shuffle if isinstance(shuffle, int) else None
            definition_list = partition_dataset(
                definition,
                self,
                partition_ratios=[1] * num_iterators,
                priority_stratification_labels=definition.data_dict['labels'],
                random_state=random_state,
                verbose=False,
                gap_tolerance=gap_tolerance
            )

            # Create iterators for each partitioned definition
            iterators = []
            for partitioned_definition in definition_list:
                iterator = self.get_iterator(partitioned_definition, window_duration, window_slide, gap_tolerance,
                                             num_windows_prefetch, "ns", label_threshold, iterator_type,
                                             window_filter_fn, shuffle, cached_windows_per_source,
                                             patient_history_fields, start_time_n, end_time_n, num_iterators=1,
                                             label_exact_match=label_exact_match, aperiodic_fill=aperiodic_fill,
                                             fill_overrides=fill_overrides, period_overrides=period_overrides)
                iterators.append(iterator)

            return iterators

        if not definition.is_validated:
            definition.validate(sdk=self, gap_tolerance=gap_tolerance, start_time=start_time_n, end_time=end_time_n)

        if definition.filtered_window_size is not None and definition.filtered_window_size != window_duration:
            warnings.warn(f"definition was filtered with window duration {definition.filtered_window_size} ns which is "
                          f"different from your requested iterator window duration {window_duration} ns. Windows will "
                          f"not be the same as the filter function's windows.")

        if definition.filtered_window_slide is not None and definition.filtered_window_slide != window_slide:
            warnings.warn(f"definition was filtered with window slide {definition.filtered_window_slide} ns which is "
                          f"different from your requested iterator window slide {window_slide} ns. Windows will "
                          f"not be the same as the filter function's windows.")

        # Reject an unsupported aperiodic_fill rule NAME up front, before any
        # measure-by-measure kind-compatibility resolution, so a typo such as
        # "carry-forward" can never be silently swallowed by the per-kind default.
        validate_fill_rule_name(aperiodic_fill, param_name="aperiodic_fill")

        number_of_values_per_window_slide = self._preflight_window_sizing(
            definition, window_duration, window_slide, period_overrides, time_units)

        if not isinstance(shuffle, bool) or shuffle:
            # Set some sensible defaults for pseudorandom yet efficient shuffle
            if cached_windows_per_source is None:
                # max(1, ...): a window slide holding more values than one block
                # (e.g. a 600 s slide at 250 Hz) floored this to 0, and
                # num_windows_prefetch = 100 * 0 = 0 degenerated the iterator to
                # one window per batch. An explicit cached_windows_per_source is
                # already asserted > 0 above; the derived one must be too.
                cached_windows_per_source = max(
                    1, self.block.block_size // number_of_values_per_window_slide)
            if num_windows_prefetch is None:
                num_windows_prefetch = max(1, 100 * cached_windows_per_source)

        else:
            # Not shuffling
            if num_windows_prefetch is None:
                num_windows_prefetch = max(
                    1, (10 * self.block.block_size) // number_of_values_per_window_slide)

        # Create appropriate iterator object based on iterator_type
        if iterator_type == 'mapped':
            iterator = MappedIterator(self, definition, window_duration, window_slide,
                                      num_windows_prefetch=num_windows_prefetch, label_threshold=label_threshold,
                                      shuffle=shuffle, max_cache_duration=max_cache_duration_per_source,
                                      patient_history_fields=patient_history_fields,
                                      label_exact_match=label_exact_match, aperiodic_fill=aperiodic_fill,
                                      fill_overrides=fill_overrides, period_overrides=period_overrides)
        elif iterator_type == 'lightmapped':
            if aperiodic_fill is not None or fill_overrides or period_overrides:
                warnings.warn("aperiodic_fill / fill_overrides / period_overrides are not applied by the "
                              "'lightmapped' iterator; it uses the numeric grid path.")
            # 'lightmapped' renders every measure through the numeric NaN sample
            # grid (return_nan_filled), which cannot represent a string measure at
            # all and mis-sizes an aperiodic one (an opaque, measure-less
            # "input array must be of size ..." from the block codec, raised deep
            # inside iteration). Fail at construction with an actionable message
            # instead. AtriumDBMapDataset follows this path as well.
            for measure_info in definition.validated_data_dict['measures']:
                signal_kind, value_type = measure_kind_of(measure_info)
                if is_string_value_type(value_type):
                    raise ValueError(
                        f"Measure {measure_info['id']} ('{measure_info['tag']}') is a string measure; "
                        f"the 'lightmapped' iterator uses the numeric NaN sample grid only and its "
                        f"values cannot be NaN-filled. Use iterator_type='mapped' (random access) or "
                        f"the default iterator_type=None, which rasterize string and aperiodic "
                        f"measures and accept aperiodic_fill / fill_overrides / period_overrides.")
                if signal_kind != SIGNAL_KIND_WAVEFORM:
                    raise ValueError(
                        f"Measure {measure_info['id']} ('{measure_info['tag']}') is an aperiodic "
                        f"('{signal_kind}') measure; the 'lightmapped' iterator uses the numeric NaN "
                        f"sample grid only and cannot rasterize it onto a nominal grid period. Use "
                        f"iterator_type='mapped' (random access) or the default iterator_type=None, "
                        f"which accept aperiodic_fill / fill_overrides / period_overrides.")
            iterator = LightMappedIterator(
                self, definition,
                window_duration, window_slide,
                label_threshold=label_threshold, shuffle=shuffle,
                patient_history_fields=patient_history_fields, label_exact_match=label_exact_match)
        elif iterator_type == 'filtered':
            if window_filter_fn is None:
                raise ValueError("window_filter_fn must be provided when iterator_type is 'filtered'")
            iterator = FilteredDatasetIterator(self, definition, window_duration, window_slide,
                                               num_windows_prefetch=num_windows_prefetch,
                                               label_threshold=label_threshold, shuffle=shuffle,
                                               max_cache_duration=max_cache_duration_per_source,
                                               window_filter_fn=window_filter_fn,
                                               patient_history_fields=patient_history_fields,
                                               label_exact_match=label_exact_match, aperiodic_fill=aperiodic_fill,
                                               fill_overrides=fill_overrides, period_overrides=period_overrides)
        elif iterator_type == "iterator":
            iterator = DatasetIterator(self, definition, window_duration, window_slide,
                                       num_windows_prefetch=num_windows_prefetch, label_threshold=label_threshold,
                                       shuffle=shuffle, max_cache_duration=max_cache_duration_per_source,
                                       patient_history_fields=patient_history_fields,
                                       label_exact_match=label_exact_match, aperiodic_fill=aperiodic_fill,
                                       fill_overrides=fill_overrides, period_overrides=period_overrides)
        else:
            raise ValueError("iterator_type must be either 'mapped', 'lightmapped','filtered' or 'iterator'")

        return iterator

    def _get_exact_interval_array_from_blocks(self, measure_id, device_id, patient_id, start, end,
                                              gap_tolerance_nano):
        block_list = self.sql_handler.select_blocks(
            int(measure_id), start_time_n=start, end_time_n=end, device_id=device_id, patient_id=patient_id)
        if len(block_list) == 0:
            return np.empty((0, 2), dtype=np.int64)

        file_id_list = [row[2] for row in condense_byte_read_list(block_list)]
        filename_dict = self.get_filename_dict(file_id_list)
        header_list = self.get_headers_from_blocks(block_list, filename_dict)

        partial_num_bytes_list = [int(header.meta_num_bytes + header.t_num_bytes) for header in header_list]
        time_read_list = [
            [row[1], row[2], row[3], row[4], partial_num_bytes]
            for row, partial_num_bytes in zip(block_list, partial_num_bytes_list)
        ]
        encoded_time_bytes = self.file_api.read_file_list(time_read_list, filename_dict)

        direct_time_decode = all(
            int(header.t_compression) == COMPRESSION_TYPES['NONE']
            and int(header.t_encoded_type) in (
                T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
                T_TYPE_TIMESTAMP_ARRAY_INT64_NANO,
            )
            for header in header_list
        )
        if direct_time_decode:
            decoded_headers = header_list
            time_arrays = []
            byte_offset = 0
            for header, partial_num_bytes in zip(decoded_headers, partial_num_bytes_list):
                time_start = byte_offset + int(header.meta_num_bytes)
                time_end = time_start + int(header.t_num_bytes)
                time_arrays.append(np.frombuffer(encoded_time_bytes[time_start:time_end], dtype=np.int64))
                byte_offset += partial_num_bytes
            time_data = None
        else:
            time_data, decoded_headers = self.block.decode_time_blocks(
                encoded_time_bytes, partial_num_bytes_list, time_type='encoded')
            time_arrays = None

        patient_ranges_by_device = clip_patient_ranges(
            self.sql_handler.get_device_time_ranges_by_patient(patient_id, end, start), start, end
        ) if patient_id is not None else None

        intervals_by_block = []
        time_offset = 0
        for block_i, (block, header) in enumerate(zip(block_list, decoded_headers)):
            period_ns = header_period_ns(header)
            time_type = int(header.t_encoded_type) if direct_time_decode else int(header.t_raw_type)
            if time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
                num_time_values = int(header.num_gaps) * 2
                if direct_time_decode:
                    block_time_data = time_arrays[block_i]
                else:
                    block_time_data = time_data[time_offset:time_offset + num_time_values]
                    time_offset += num_time_values
                intervals = intervals_from_gap_data(
                    int(header.start_n), block_time_data, int(header.num_vals), period_ns)
            elif time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
                num_time_values = int(header.num_vals)
                if direct_time_decode:
                    block_time_data = time_arrays[block_i]
                else:
                    block_time_data = time_data[time_offset:time_offset + num_time_values]
                    time_offset += num_time_values
                intervals = intervals_from_timestamps(block_time_data, period_ns)
            else:
                raise ValueError(f"Unsupported decoded time type {time_type}")

            clipped = clip_intervals_to_source_ranges(
                intervals, block[2], start, end, patient_ranges_by_device)
            if clipped.size:
                intervals_by_block.append(clipped)

        return intervals_union_list(intervals_by_block, gap_tolerance_nano=gap_tolerance_nano)

    def get_interval_array(self, measure_id=None, device_id=None, patient_id=None,
                           gap_tolerance_nano: int = 0, start=None, end=None, measure_tag=None,
                           freq=None, units=None, freq_units=None, device_tag=None, mrn: str=None,
                           exact: bool = False):
        """
        .. _get_interval_array_label:

        Returns a 2D array representing the availability of a specified measure (signal) and a specified source
        (device id or patient id). Each row of the 2D array output represents a continuous interval of available
        data while the first and second columns represent the start epoch and end epoch of that interval
        respectively.

        **Coarse presence for aperiodic kinds.** For ``waveform`` measures the interval array is a tight,
        near-exact map of where continuous data exists. For aperiodic ``signal_kind`` values
        (``sample``, ``event``, ``state``) the interval index is a deliberately *coarse presence* map --
        it answers "are there readings/events roughly in this window" (the underlying writes use a widened
        gap tolerance so irregular arrivals do not flood the index). For precise per-sample or per-event
        timing on those kinds, read the actual stored timestamps via
        :ref:`get_data <get_data_label>` / :ref:`get_string_data <get_string_data_label>` instead of relying
        on this method. Pass ``exact=True`` for a waveform measure to reconstruct continuous coverage from the
        stored block time payloads instead of the interval index. Pass ``gap_tolerance_nano`` to control how
        aggressively adjacent intervals are merged.

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
        :param str mrn: Medical record number for the patient. Exclusive with patient_id. An int can be provided, but will be converted and stored as a string.
        :param bool exact: If True for a local waveform measure, reconstruct intervals from block time data rather
            than using the interval index. Aperiodic measures still return the coarse presence index, because their
            exact timing is represented as samples/events/states rather than continuous waveform coverage.
        :rtype: numpy.ndarray
        :returns: A 2D array representing the availability of a specified measure.

        """

        if device_id is None and device_tag is not None:
            device_id = self.get_device_id(device_tag)

        if patient_id is None and mrn is not None:
            patient_id = self.get_patient_id(mrn)

        # Check if the metadata connection type is API
        if self.metadata_connection_type == "api":
            if exact:
                raise NotImplementedError("exact interval reconstruction is only supported in local mode.")
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

        measure_info = self.get_measure_info(measure_id)
        if exact and measure_info is not None and measure_info.get('signal_kind') == SIGNAL_KIND_WAVEFORM:
            return self._get_exact_interval_array_from_blocks(
                measure_id, device_id, patient_id, start, end, gap_tolerance_nano)

        # Query the database for intervals based on the given parameters
        interval_result = self.sql_handler.select_intervals(
            measure_id, start_time_n=start, end_time_n=end, device_id=device_id, patient_id=patient_id)

        # Initialize an empty list to store the final intervals
        arr = []
        # Iterate through the sorted interval results
        for row in interval_result:
            # if the start is greater than or equal to the end_time of this interval skip this interval
            # also if the end time is less than or equal to the current intervals start_time skip the interval
            if (start and start >= row[4]) or (end and end <= row[3]) or (row[3] >= row[4]):
                continue

            # If the final intervals list is not empty and the difference between the current interval's start time
            # and the previous interval's end time is less than or equal to the gap tolerance, update the end time
            # of the previous interval
            cur_interval_start = row[3] if start is None else max(row[3], start)
            cur_interval_end = row[4] if end is None else min(row[4], end)
            if arr and cur_interval_start - arr[-1][-1] <= gap_tolerance_nano:
                arr[-1][-1] = max(cur_interval_end, arr[-1][-1])
            # Otherwise, add a new interval to the final intervals list
            else:
                arr.append([cur_interval_start, cur_interval_end])

        # Convert the final intervals list to a numpy array with int64 data type
        return np.array(arr, dtype=np.int64)

    def optimize_interval_index(self, gap_tolerance: int = None, *, measure_id: int = None,
                                device_id: int = None, batch_size: int = 10_000):
        """Compact legacy interval-index rows using the current gap policy.

        Older writers and ``interval_index_mode="fast"`` append one row for each
        interval instead of merging adjacent rows. This online maintenance pass
        coalesces those rows while keeping only one measure/device stream page in
        memory. A page contains at most ``batch_size`` rows (10,000 by default),
        and each page has its own transaction, so it is suitable for indexes with
        hundreds of millions of rows.

        When ``gap_tolerance`` is omitted, the optimizer uses the period-based
        portion of the current smart default for each measure
        (:func:`choose_interval_gap_tolerance`). Pass an integer number of
        nanoseconds to apply an explicit policy exactly, including ``0`` to merge
        only overlapping or touching rows. The write-time aperiodic widening is
        intentionally not guessed retroactively: it depends on the median spacing
        in each original write, information that a legacy interval row does not
        preserve. Supply the intended explicit tolerance when that distinction
        matters.

        The operation is safe to run during ingestion. It only deletes row IDs
        selected in its own short transaction; a concurrently inserted row is never
        deleted. Rows inserted behind an already-scanned cursor are left for a
        subsequent pass, making concurrent operation eventually convergent rather
        than requiring an ingestion pause.

        :param int gap_tolerance: Optional explicit gap tolerance in nanoseconds.
        :param int measure_id: Optionally optimize just one measure.
        :param int device_id: Optionally optimize just one device of ``measure_id``.
        :param int batch_size: Maximum interval rows read and changed per transaction.
        :return: A mapping with ``pairs_processed``, ``rows_examined`` and
            ``rows_merged`` counters.
        :rtype: dict
        """
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not supported for interval-index optimization.")
        if device_id is not None and measure_id is None:
            raise ValueError("device_id requires measure_id")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if gap_tolerance is not None:
            try:
                normalized_gap_tolerance = int(gap_tolerance)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("gap_tolerance must be a non-negative integer number of nanoseconds") from error
            if isinstance(gap_tolerance, bool) or normalized_gap_tolerance != gap_tolerance or normalized_gap_tolerance < 0:
                raise ValueError("gap_tolerance must be a non-negative integer number of nanoseconds")
            gap_tolerance = normalized_gap_tolerance

        # The metadata table is tiny compared with interval_index. Building this
        # lookup once keeps the SQL handler generic while its actual scan stays
        # bounded by batch_size even for very large interval tables.
        tolerances = {}
        for row in self.sql_handler.select_all_measures():
            current_measure_id, _, _, freq_nhz, stored_period_ns = row[:5]
            if measure_id is not None and int(current_measure_id) != int(measure_id):
                continue
            if gap_tolerance is None:
                period_ns = stored_period_ns
                if period_ns is None and freq_nhz is not None and int(freq_nhz) > 0:
                    period_ns = 10 ** 18 // int(freq_nhz)
                tolerances[int(current_measure_id)] = choose_interval_gap_tolerance(period_ns)
            else:
                tolerances[int(current_measure_id)] = int(gap_tolerance)

        if measure_id is not None and int(measure_id) not in tolerances:
            raise ValueError(f"Unknown measure_id={measure_id}")
        return self.sql_handler.optimize_interval_index(
            tolerances, measure_id=measure_id, device_id=device_id, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Beds and sources
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # API mode: HTTP transport, token refresh and shutdown
    # ------------------------------------------------------------------ #
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

        # if the user is using a .env file to store the token, persist the new
        # one there for future processes (self.token is authoritative in this one)
        if self.dot_env_loaded:
            set_key("./.env", "ATRIUMDB_API_TOKEN", token_data['access_token'])

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
        elif (self.metadata_connection_type == "mariadb" or self.metadata_connection_type == "mysql") and self.sql_handler.connection_manager is not None:
            self.sql_handler.connection_manager.close_connection()

    # ------------------------------------------------------------------ #
    # Low-level block/TSC file access, settings and remaining internals
    # ------------------------------------------------------------------ #
    def get_filename_dict(self, file_id_list):
        if self.metadata_connection_type == "api":
            raise ValueError("This function is only meant to work in local mode.")

        result_dict = {}

        # Query file index table for file_id, filename pairs
        for row in self.sql_handler.select_files(file_id_list):
            # Add them to a dictionary {file_id: filename}
            result_dict[row[0]] = row[1]

        return result_dict

    def _get_all_settings(self):
        settings = self.sql_handler.select_all_settings()
        return {setting[0]: setting[1] for setting in settings}

    def _merge_conflict_policy(self):
        """How block merging resolves duplicate timestamps between a new write
        and existing data, from the dataset's 'overwrite' setting:

        * ``'overwrite'`` (and the legacy default ``'ignore'``, which historically
          meant "skip overlap handling entirely") - the new write's values win.
        * ``'protect'`` - the existing data's values win.
        * ``'error'`` - refuse (raise) a merge whose write shares timestamps with
          the block it would merge into.

        The policy is enforced where deduplication happens on write: writes smaller
        than one block that merge with an existing block, and duplicate pushes within
        one buffer flush. Overlapping writes of a full block or more never merge, so
        both copies are stored under 'overwrite'/'protect' -- accepted, because write
        speed is the priority and duplicates are expected at live ingest. The same
        policy then governs the READ-side collapse
        (``get_data(..., allow_duplicates=False)``), so a read resolves a surviving
        duplicate exactly as a write would have. Under 'error',
        :meth:`_report_undeduplicated_overlap` refuses such a write outright."""
        setting = getattr(self, 'settings_dict', None) and self.settings_dict.get(OVERWRITE_SETTING_NAME)
        if setting in ('protect', 'error'):
            return setting
        return 'overwrite'

    def get_data_from_tsc_file(self, file_path, analog=True, time_type=1, sort=True, allow_duplicates=True):
        if self.metadata_connection_type == "api":
            raise NotImplementedError("API mode is not yet supported for this function.")

        encoded_bytes = self.file_api.read_from_filepath(file_path)

        r_times, r_values, headers = self.block.decode_block_from_bytes_alone(
            encoded_bytes, analog=analog, time_type=time_type)

        # Sort the data based on the timestamps if sort is true
        if sort and time_type == 1:
            r_times, r_values = sort_data(r_times, r_values, headers, 0, (2**63) - 1, allow_duplicates,
                                          duplicate_keep=self._duplicate_keep())

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
            r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates,
                                          duplicate_keep=self._duplicate_keep())

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


    def get_headers(self, measure_id: int = None, start_time_n: int = None, end_time_n: int = None,
                    device_id: int = None, patient_id=None, block_info=None,
                    time_units: str = None, measure_tag: str = None,
                    freq: Union[int, float] = None, units: str = None, freq_units: str = None,
                    device_tag: str = None, mrn: str = None):
        """
        .. _get_headers_label:

        Read only the block headers for a query, without decoding the signal itself.

        Takes the same signal / time / source arguments as :meth:`get_data` and returns
        exactly the header list that call would have returned as its first element --
        but reads only each block's fixed-size header off disk instead of the whole
        block, so the cost does not scale with how many samples the range holds. Use it
        to inspect the shape of stored data (block boundaries, per-block time ranges,
        sample counts, encodings) when the values themselves are not needed.

        If measure_id is None, measure_tag along with freq and units must not be None, and vice versa.
        Similarly, if device_id is None, device_tag must not be None, and if patient_id is None, mrn must not be None.

        >>> headers = sdk.get_headers(measure_id, start_time_n, end_time_n, device_id=device_id)
        >>> sum(header.num_vals for header in headers)   # samples in range, nothing decoded
        20000

        ``BlockMetadata`` is a ``ctypes.Structure``, so ``==`` on it compares identity
        rather than contents; compare the fields if you need value equality.

        :param int measure_id: The measure identifier. If None, measure_tag must be provided.
        :param int start_time_n: The start epoch in nanoseconds of the data you would like to query.
        :param int end_time_n: The end epoch in nanoseconds. The end time is not inclusive.
        :param int device_id: The device identifier. If None, device_tag must be provided.
        :param int patient_id: The patient identifier. If None, mrn must be provided.
        :param block_info: A precomputed ``{'block_list': ..., 'filename_dict': ...}`` pair,
            as returned by the block-lookup helpers, used to skip the metadata query when
            the caller has already resolved the blocks.
        :param str time_units: Unit that `start_time_n` and `end_time_n` are given in.
            Options: ["s", "ms", "us", "ns"], default "ns".
        :param str measure_tag: A short string identifying the signal. Required if measure_id is None.
        :param freq: The sample frequency of the signal. Helpful with measure_tag.
        :param str units: The units of the signal. Helpful with measure_tag.
        :param str freq_units: Units for frequency. Options: ["nHz", "uHz", "mHz", "Hz", "kHz", "MHz"] default "nHz".
        :param str device_tag: A string identifying the device. Exclusive with device_id.
        :param str mrn: Medical record number for the patient. Exclusive with patient_id. An int can be provided, but will be converted and stored as a string.

        :rtype: List[BlockMetadata]
        :returns: One header per block overlapping the query, ordered by file and byte
            offset. Empty if the range holds no blocks.
        :raises NotImplementedError: In API mode -- this is a local-mode method.
        :raises ValueError: If `time_units` is not one of the accepted units.
        """
        time_units, start_time_n, end_time_n, device_id, patient_id = self._resolve_read_range(
            start_time_n, end_time_n, time_units, device_id, device_tag, patient_id, mrn)

        measure_id = self._resolve_read_measure_id(measure_id, measure_tag, freq, units, freq_units)

        if self.mode == "api":
            raise NotImplementedError("get_headers is unavailable for API mode")

        device_id = int(device_id) if device_id is not None else device_id

        block_list, filename_dict, _ = self._select_read_blocks(
            measure_id, device_id, patient_id, start_time_n, end_time_n, block_info)

        # if no matching block ids
        if len(block_list) == 0:
            return []

        # Get headers from blocks without reading the actual data
        return self.get_headers_from_blocks(block_list, filename_dict)


    def get_headers_from_blocks(self, block_list, filename_dict):
        """
        Retrieve only the headers from blocks without decoding the actual wave data.

        This method reads only the header portion of the specified blocks, providing
        metadata information without the overhead of decoding time and value data.

        :param list block_list: List of blocks to read headers from.
        :param dict filename_dict: Dictionary containing file information.
        :return: List of block headers.
        :rtype: List[BlockMetadata]
        """
        if self.metadata_connection_type == "api":
            raise ValueError("This function is only meant to work in local mode.")

        header_size = sizeof(BlockMetadata)

        header_read_list = [[row[1], row[2], row[3], row[4], header_size] for row in block_list]

        encoded_header_bytes = self.file_api.read_file_list(header_read_list, filename_dict)

        num_headers = len(block_list)
        byte_start_array = np.arange(0, num_headers * header_size, header_size, dtype=np.uint64)

        headers = self.block.decode_headers(encoded_header_bytes, byte_start_array)

        return headers
