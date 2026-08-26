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

import time
from contextlib import contextmanager
import threading

import mariadb
from mariadb import ProgrammingError

from typing import List, Dict, Tuple

from atriumdb.adb_functions import allowed_interval_index_modes
from atriumdb.sql_handler.maria.maria_functions import maria_select_measure_from_triplet_query, \
    maria_select_measure_from_id, \
    maria_insert_file_index_query, maria_insert_block_query, \
    maria_select_block_by_id, maria_select_block_by_values, \
    mariadb_setting_insert_query, \
    maria_insert_ignore_device_encounter_query, maria_insert_ignore_encounter_query, maria_insert_ignore_patient_query, \
    maria_delete_block_query, maria_insert_interval_index_query
from atriumdb.sql_handler.maria.maria_tables import mariadb_measure_create_query, \
    maria_file_index_create_query, maria_block_index_create_query, maria_interval_index_create_query, \
    maria_settings_create_query, maria_device_encounter_create_query, maria_source_create_query, \
    maria_institution_create_query, maria_unit_create_query, maria_bed_create_query, maria_patient_create_query, \
    maria_encounter_create_query, mariadb_device_create_query, maria_insert_adb_source, \
    mariadb_log_hl7_adt_create_query, mariadb_current_census_view, mariadb_device_patient_table, \
    maria_encounter_insert_trigger, maria_encounter_update_trigger, maria_encounter_delete_trigger, \
    maria_insert_interval_stored_procedure, maria_insert_interval_union_stored_procedure, \
    maria_patient_history_create_query, mariadb_label_set_create_query, \
    mariadb_label_create_query, mariadb_label_source_create_query
from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler, block_insert_tuples, interval_insert_tuples
import logging

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 3306


def maria_connection_args(connection_params: dict):
    """Return ``MariaDBHandler``'s positional connection arguments in order."""
    return tuple(connection_params[key]
                 for key in ('host', 'user', 'password', 'database', 'port'))


class SingleConnectionManager:
    def __init__(self, validation_interval=500, **conn_args):
        self._connection = None
        self._conn_args = conn_args
        self._lock = threading.RLock()
        self.__last_used = 0
        self._validation_interval = validation_interval
        self._borrowed = False

    def _create_connection(self):
        attempts = 0
        while attempts < 5:
            try:
                self._connection = mariadb.connect(**self._conn_args)
                return
            except mariadb.Error as e:
                attempts += 1
                if attempts >= 5:
                    raise e
                time.sleep(1)

    def get_connection(self):
        with self._lock:
            # If the connection is already borrowed, raise Error
            if self._borrowed:
                raise ValueError("The connection is already borrowed, please release it before requesting it again.")

            # Check and create a new connection if needed
            if self._connection is None:
                self._create_connection()
            else:
                dt = (time.perf_counter_ns() - self.__last_used) / 1_000_000
                if dt > self._validation_interval:
                    try:
                        self._connection.ping()
                    except mariadb.Error:
                        self._create_connection()
            self.__last_used = time.perf_counter_ns()
            self._borrowed = True  # Mark the connection as borrowed
            return self._connection

    def release_connection(self):
        with self._lock:
            self._borrowed = False  # Mark the connection as available

    def close_connection(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._borrowed = False


class MariaDBHandler(SQLHandler):
    _MISSING_COLUMN_PHRASE = "unknown column"
    _MISSING_TABLE_ERRORS = (ProgrammingError,)
    _MISSING_TABLE_PHRASES = ("Table", "doesn't exist")

    def __init__(self, host: str, user: str, password: str, database: str, port: int = None, no_pool=False,
                 validation_interval=500):
        self.host = host
        self.user = user
        self.password = password
        self.port = DEFAULT_PORT if port is None else port
        self.database = database

        self.connection_params = {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
        }
        self.no_pool = no_pool
        self.connection_manager = None if no_pool else SingleConnectionManager(
            validation_interval=validation_interval, **self.connection_params)

        # Whether the dataset has the insert_interval_union stored procedure
        # (created by create_schema since the smart-write-defaults work). Datasets
        # created before it fall back to the legacy insert_interval procedure the
        # first time the union procedure is found missing.
        self._interval_union_proc_available = True

    def _call_insert_interval(self, cursor, interval: Dict, gap_tolerance: int):
        """Insert one interval row in "merge" mode. Prefers insert_interval_union
        (unions ALL rows bridged within gap_tolerance - same semantics as the
        SQLite handler, with the identical append fast path); falls back to the
        legacy insert_interval (merges with at most one row) on datasets that
        predate the union procedure. A missing-procedure error is a statement
        level error in MariaDB, so retrying within the same transaction is safe."""
        params = (interval["measure_id"], interval["device_id"], interval["start_time_n"],
                  interval["end_time_n"], gap_tolerance)
        if self._interval_union_proc_available:
            try:
                cursor.callproc("insert_interval_union", params)
                return
            except (ProgrammingError, mariadb.OperationalError) as e:
                # 1305 = PROCEDURE does not exist (the connector surfaces it as
                # OperationalError); anything else is a real failure.
                if getattr(e, "errno", None) != 1305:
                    raise
                self._interval_union_proc_available = False
        cursor.callproc("insert_interval", params)

    def _interval_optimizer_lock_clause(self) -> str:
        """Lock a maintenance page so merge-mode ingestion cannot rewrite it.

        The optimizer commits after each bounded page, keeping these locks short
        while still preventing it from deleting or overwriting a row a concurrent
        ``insert_interval_union`` call is updating.
        """
        return " FOR UPDATE"

    def maria_connect(self):
        return mariadb.connect(**self.connection_params)

    def maria_connect_no_db(self, db_name=None):
        conn = mariadb.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=db_name)

        return conn

    def check_database_exists(self):
        """Check if the specified database exists."""
        conn = self.maria_connect_no_db()
        cursor = conn.cursor()
        cursor.execute(f"SHOW DATABASES LIKE '{self.database}'")
        exists = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return exists

    @contextmanager
    def maria_db_connection(self, begin=False):
        conn = self.maria_connect() if self.no_pool else self.connection_manager.get_connection()
        if conn is None:
            # This error shouldn't get hit, because the connection manager / mariadb
            # should handle errors before this line.
            raise ValueError("Something went wrong with the MariaDB connection.")

        cursor = conn.cursor()

        try:
            if begin:
                conn.begin()
            yield conn, cursor
            conn.commit()
        except mariadb.Error as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            if self.no_pool:
                conn.close()
            else:
                self.connection_manager.release_connection()

    def connection(self, begin=False):
        # Overrides the inherited class method.
        return self.maria_db_connection(begin=begin)

    def create_schema(self):
        conn = self.maria_connect_no_db()
        cursor = conn.cursor()

        # Create Schema
        cursor.execute("CREATE DATABASE IF NOT EXISTS `{}`".format(self.database))
        cursor.close()
        conn.change_user(self.user, self.password, self.database)
        cursor = conn.cursor()

        # Create Tables
        cursor.execute(maria_source_create_query)
        cursor.execute(maria_institution_create_query)
        cursor.execute(maria_unit_create_query)
        cursor.execute(maria_bed_create_query)

        cursor.execute(maria_patient_create_query)
        cursor.execute(maria_patient_history_create_query)
        cursor.execute(maria_encounter_create_query)

        cursor.execute(mariadb_measure_create_query)
        cursor.execute(mariadb_device_create_query)
        cursor.execute(maria_file_index_create_query)
        cursor.execute(maria_block_index_create_query)
        cursor.execute(maria_interval_index_create_query)
        cursor.execute(maria_settings_create_query)

        cursor.execute(maria_device_encounter_create_query)
        cursor.execute(mariadb_log_hl7_adt_create_query)
        cursor.execute(mariadb_device_patient_table)

        cursor.execute(mariadb_label_set_create_query)
        cursor.execute(mariadb_label_source_create_query)
        cursor.execute(mariadb_label_create_query)

        # Create Views
        cursor.execute(mariadb_current_census_view)

        # Insert Default Values
        cursor.execute(maria_insert_adb_source)

        # Triggers
        cursor.execute(maria_encounter_insert_trigger)
        cursor.execute(maria_encounter_update_trigger)
        cursor.execute(maria_encounter_delete_trigger)

        # Stored Procedures
        cursor.execute(maria_insert_interval_stored_procedure)
        cursor.execute(maria_insert_interval_union_stored_procedure)

        conn.commit()
        cursor.close()
        conn.close()

        # Stamp the dataset a brand-new schema produces as current, so opening it
        # right after (even with the default auto_upgrade=False) never sees a
        # pending upgrade -- see SQLHandler.pending_schema_upgrades.
        self.record_dataset_schema_version()

    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = ? 
            AND TABLE_NAME = ? 
            AND COLUMN_NAME = ?
        """, (self.database, table_name, column_name))
        return cursor.fetchone()[0] > 0

    def update_measure_schema(self):
        """Add the additive nullable measure columns if they do not exist.

        Idempotent, additive migration mirroring the SQLite handler:
        ``period_ns`` plus the measure-kind columns ``signal_kind`` and
        ``value_type``. NULLs are read-time defaulted by the SDK
        (``signal_kind`` -> ``waveform``, ``value_type`` -> ``numeric``), so
        existing rows need no backfill for correctness. Returns True if any
        column was added."""
        changed = False
        with self.connection() as (conn, cursor):
            if not self._column_exists(cursor, 'measure', 'period_ns'):
                cursor.execute("ALTER TABLE measure ADD COLUMN period_ns BIGINT NULL")
                changed = True
            if not self._column_exists(cursor, 'measure', 'signal_kind'):
                cursor.execute("ALTER TABLE measure ADD COLUMN signal_kind VARCHAR(16) NULL")
                changed = True
            if not self._column_exists(cursor, 'measure', 'value_type'):
                cursor.execute("ALTER TABLE measure ADD COLUMN value_type VARCHAR(16) NULL")
                changed = True
            if changed:
                conn.commit()
        return changed

    def check_mrn_column_is_text(self) -> bool:
        """Check if the mrn column in the patient table is VARCHAR/TEXT. Returns True if it is."""
        with self.connection() as (conn, cursor):
            cursor.execute("""
                SELECT DATA_TYPE 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = ? 
                AND TABLE_NAME = 'patient' 
                AND COLUMN_NAME = 'mrn'
            """, (self.database,))
            result = cursor.fetchone()
            if result is None:
                return False
            return result[0].upper() in ('VARCHAR', 'TEXT', 'CHAR')

    def upgrade_mrn_schema(self):
        """Upgrade the patient table mrn column from INT to VARCHAR if needed."""
        if self.check_mrn_column_is_text():
            return False  # Already VARCHAR/TEXT, no upgrade needed

        with self.connection() as (conn, cursor):
            cursor.execute("""
                ALTER TABLE patient 
                MODIFY COLUMN mrn VARCHAR(255) NULL
            """)
            conn.commit()
            return True

    def select_all_measures(self):
        try:
            with self.maria_db_connection() as (conn, cursor):
                cursor.execute("""
                    SELECT id, tag, name, freq_nhz, period_ns, code, unit, unit_label, unit_code, source_id, signal_kind, value_type
                    FROM measure
                """)
                rows = cursor.fetchall()
            return rows
        except mariadb.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def select_all_patients(self):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute("SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height  FROM patient")
            rows = cursor.fetchall()
        return rows

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None,
                       measure_id=None, code: str = None, unit_label: str = None, unit_code: str = None,
                       source_id: int = None, period_ns: int = None, signal_kind: str = None,
                       value_type: str = None):
        units = DEFAULT_UNITS if units is None else units

        try:
            with self.connection() as (conn, cursor):
                cursor.execute(
                    "INSERT IGNORE INTO measure (id, tag, freq_nhz, period_ns, unit, name, code, unit_label, unit_code, source_id, signal_kind, value_type) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (measure_id, measure_tag, freq_nhz, period_ns, units, measure_name, code, unit_label, unit_code,
                     source_id, signal_kind, value_type))
                conn.commit()
                return cursor.lastrowid
        except mariadb.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units

        try:
            with self.maria_db_connection() as (conn, cursor):
                if measure_id is not None:
                    cursor.execute(maria_select_measure_from_id, (measure_id,))
                else:
                    cursor.execute(maria_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
                row = cursor.fetchone()

            return row
        except mariadb.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def set_string_dict_watermark(self, measure_id: int, vocabulary_size: int):
        """Raise this measure's recorded string-vocabulary size (see
        ``SQLHandler.set_string_dict_watermark``). One statement, so concurrent
        writers cannot interleave a read and a write and lower the mark; GREATEST
        makes it monotonic."""
        name = self._string_dict_size_setting_name(measure_id)
        value = str(int(vocabulary_size))
        with self.maria_db_connection(begin=True) as (conn, cursor):
            cursor.execute(
                "INSERT INTO setting (name, value) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE value = "
                "GREATEST(CAST(setting.value AS UNSIGNED), CAST(VALUES(value) AS UNSIGNED))",
                (name, value))

    def _upsert_setting(self, name: str, value: str):
        """Unconditional single-statement upsert for one `setting` row (see
        ``SQLHandler._upsert_setting``) -- unlike ``set_string_dict_watermark``
        above, there is no monotonic comparison; the new value always wins."""
        with self.maria_db_connection(begin=True) as (conn, cursor):
            cursor.execute(
                "INSERT INTO setting (name, value) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (name, value))

    def interval_union_procedure_current(self) -> bool:
        """Read-only check for the insert_interval_union stored procedure, so a
        caller can learn a dataset needs :meth:`ensure_interval_union_procedure`
        without running it."""
        with self.connection() as (conn, cursor):
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.ROUTINES "
                "WHERE ROUTINE_SCHEMA = ? AND ROUTINE_TYPE = 'PROCEDURE' AND ROUTINE_NAME = ?",
                (self.database, "insert_interval_union"))
            return cursor.fetchone()[0] > 0

    def ensure_interval_union_procedure(self) -> bool:
        """Create insert_interval_union if this dataset predates it -- create_schema
        only runs at dataset creation (see maria_tables.py), so an existing dataset
        never gets the procedure added any other way. `CREATE PROCEDURE IF NOT
        EXISTS` already makes this idempotent on its own; the check up front only
        exists so the return value tells the caller whether anything changed.
        Also marks the procedure available on THIS handler instance, in case an
        earlier call on it had already flipped _interval_union_proc_available to
        False after finding it missing."""
        was_current = self.interval_union_procedure_current()
        if not was_current:
            with self.connection() as (conn, cursor):
                cursor.execute(maria_insert_interval_union_stored_procedure)
                conn.commit()
        self._interval_union_proc_available = True
        return not was_current

    def insert_device(self, device_tag: str, device_name: str = None, device_id=None, manufacturer: str = None,
                      model: str = None, device_type: str = None, bed_id: int = None, source_id: int = None):
        device_type = "static" if device_type is None else device_type
        source_id = 1 if source_id is None else source_id
        with self.connection() as (conn, cursor):
            cursor.execute(
                "INSERT IGNORE INTO device (id, tag, name, manufacturer, model, type, bed_id, source_id) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?);",
                (device_id, device_tag, device_name, manufacturer, model, device_type, bed_id, source_id))
            conn.commit()

            return cursor.lastrowid

    def _insert_intervals(self, cursor, interval_data: List[Dict], interval_index_mode, gap_tolerance: int = 0):
        """Insert interval rows using the given interval_index_mode. "fast" appends raw
        rows; "merge" defers to the insert-interval stored procedure, which unions each
        new interval with the existing rows it bridges; "disable" does nothing."""
        if interval_index_mode == "fast":
            cursor.executemany(maria_insert_interval_index_query, interval_insert_tuples(interval_data))

        elif interval_index_mode == "merge":
            for interval in interval_data:
                self._call_insert_interval(cursor, interval, gap_tolerance)
        elif interval_index_mode == "disable":
            # Do Nothing
            pass
        else:
            raise ValueError(f"interval_index_mode must be one of {allowed_interval_index_modes}")

    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode, gap_tolerance: int = 0):
        # default to merge mode
        interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode

        with self.maria_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(maria_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            cursor.executemany(maria_insert_block_query, block_insert_tuples(block_data, file_id))

            # insert into interval_index
            self._insert_intervals(cursor, interval_data, interval_index_mode, gap_tolerance)

    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int], gap_tolerance: int = 0):
        with self.maria_db_connection(begin=True) as (conn, cursor):
            # insert/update file data
            for file_path, (block_data, interval_data) in file_data.items():
                # insert file_path into file_index and get id
                cursor.execute(maria_insert_file_index_query, (file_path,))
                file_id = cursor.lastrowid

                # insert into block_index
                cursor.executemany(maria_insert_block_query, block_insert_tuples(block_data, file_id))

                # insert into interval_index
                for interval in interval_data:
                    self._call_insert_interval(cursor, interval, gap_tolerance)

            # delete old block data
            cursor.executemany(maria_delete_block_query, [(block_id,) for block_id in block_ids_to_delete])

            # delete old file data (will delete later)
            # cursor.executemany(maria_delete_file_query, [(file_id,) for file_id in file_ids_to_delete])

    def insert_merged_block_data(self, file_path: str, block_data: List[Dict], old_block: tuple, interval_data: List[Dict],
                                 interval_index_mode, gap_tolerance: int = 0):
        # default to merge mode
        interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode

        with self.maria_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(maria_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            cursor.executemany(maria_insert_block_query, block_insert_tuples(block_data, file_id))

            # insert into interval_index
            self._insert_intervals(cursor, interval_data, interval_index_mode, gap_tolerance)

            return self._delete_merged_block(cursor, old_block)

    def select_block(self, block_id: int = None, measure_id: int = None, device_id: int = None, file_id: int = None,
                     start_byte: int = None, num_bytes: int = None, start_time_n: int = None, end_time_n: int = None,
                     num_values: int = None):
        with self.maria_db_connection() as (conn, cursor):
            if block_id is not None:
                cursor.execute(maria_select_block_by_id, (block_id,))
            else:
                cursor.execute(maria_select_block_by_values, (measure_id, device_id, file_id, start_byte, num_bytes,
                                                              start_time_n, end_time_n, num_values))
            row = cursor.fetchone()
        return row

    def insert_setting(self, setting_name: str, setting_value: str):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(mariadb_setting_insert_query, (setting_name, setting_value))

    def insert_patient(self, patient_id=None, mrn=None, gender=None, dob=None, first_name=None, middle_name=None,
                       last_name=None, first_seen=None, last_updated=None, source_id=1, weight=None, height=None):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_patient_query,
                           (patient_id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated,
                            source_id, weight, height))
            return cursor.lastrowid

    def insert_encounter(self, patient_id, bed_id, start_time, end_time=None, source_id=1, visit_number=None,
                         last_updated=None):
        last_updated = time.time_ns() if last_updated is None else last_updated
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_encounter_query,
                           (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated))
            return cursor.lastrowid

    def insert_device_encounter(self, device_id, encounter_id, start_time, end_time=None, source_id=1):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_device_encounter_query,
                           (device_id, encounter_id, start_time, end_time, source_id))
            return cursor.lastrowid

    def select_encounters(self, patient_id_list: List[int] = None, mrn_list: List[str] = None, start_time: int = None,
                          end_time: int = None):
        assert (patient_id_list is None) != (mrn_list is None), "Either patient_id_list or mrn_list must be provided, but not both"
        # An empty filter list names no patients, so no encounters. Building
        # `IN ()` instead is a syntax error on MariaDB.
        if patient_id_list is not None and len(patient_id_list) == 0:
            return []
        if mrn_list is not None and len(mrn_list) == 0:
            return []

        arg_tuple = ()
        maria_select_encounter_query = \
            "SELECT encounter.id, encounter.patient_id, encounter.bed_id, encounter.start_time, encounter.end_time, " \
            "encounter.source_id, encounter.visit_number, encounter.last_updated FROM encounter"
        if patient_id_list is not None:
            maria_select_encounter_query += \
                " INNER JOIN patient ON encounter.patient_id = patient.id WHERE encounter.patient_id IN ({})".format(
                    ','.join(['?'] * len(patient_id_list)))
            arg_tuple += tuple(patient_id_list)
        else:
            maria_select_encounter_query += \
                " INNER JOIN patient ON encounter.patient_id = patient.id WHERE patient.mrn IN ({})".format(
                    ','.join(['?'] * len(mrn_list)))

            arg_tuple += tuple(mrn_list)
        if start_time is not None:
            maria_select_encounter_query += " AND encounter.end_time > ?"
            arg_tuple += (start_time,)
        if end_time is not None:
            maria_select_encounter_query += " AND encounter.start_time < ?"
            arg_tuple += (end_time,)
        maria_select_encounter_query += " ORDER BY encounter.id ASC"

        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_select_encounter_query, arg_tuple)
            return cursor.fetchall()

    def select_all_patients_in_list(self, patient_id_list: List[int] = None, mrn_list: List[str] = None):
        if patient_id_list is not None:
            # An empty filter list names no rows. Building `IN ()` instead is a
            # syntax error on MariaDB and a silent "matches nothing" on SQLite.
            if len(patient_id_list) == 0:
                return []
            placeholders = ', '.join(['?'] * len(patient_id_list))
            maria_select_patients_by_id_list = f"SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient WHERE id IN ({placeholders})"
        elif mrn_list is not None:
            # An empty filter list names no rows. Building `IN ()` instead is a
            # syntax error on MariaDB and a silent "matches nothing" on SQLite.
            if len(mrn_list) == 0:
                return []
            patient_id_list = mrn_list
            placeholders = ', '.join(['?'] * len(patient_id_list))
            maria_select_patients_by_id_list = f"SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height  FROM patient WHERE mrn IN ({placeholders})"
        else:
            maria_select_patients_by_id_list = "SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height  FROM patient"
            patient_id_list = tuple()
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_patients_by_id_list, patient_id_list)
            rows = cursor.fetchall()
        return rows
