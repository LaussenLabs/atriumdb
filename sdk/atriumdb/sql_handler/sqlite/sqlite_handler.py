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

import sqlite3
from pathlib import Path
from typing import Union, List, Dict, Tuple
from contextlib import contextmanager
import time

from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler, block_insert_tuples, interval_insert_tuples
from atriumdb.sql_handler.sqlite.sqlite_functions import (
    sqlite_select_measure_from_triplet_query, sqlite_select_measure_from_id_query, \
    sqlite_insert_file_index_query, \
    sqlite_insert_block_query, sqlite_insert_interval_index_query, \
    sqlite_select_block_by_id, sqlite_select_block_by_values, \
    sqlite_setting_insert_query, \
    sqlite_insert_ignore_patient_query, sqlite_insert_ignore_encounter_query, \
    sqlite_insert_ignore_device_encounter_query, sqlite_delete_block_query, \
    sqlite_interval_exists_query)
from atriumdb.sql_handler.sqlite.sqlite_tables import sqlite_measure_create_query, \
    sqlite_file_index_create_query, sqlite_block_index_create_query, \
    sqlite_interval_index_create_query, sqlite_settings_create_query, \
    sqlite_block_index_idx_query, sqlite_interval_index_idx_query, sqlite_device_encounter_create_query, \
    sqlite_device_encounter_device_id_create_index, sqlite_device_encounter_encounter_id_create_index, \
    sqlite_device_encounter_source_id_create_index, sqlite_source_create_query, sqlite_institution_create_query, \
    sqlite_unit_create_query, sqlite_unit_institution_id_create_index, sqlite_bed_create_query, \
    sqlite_bed_unit_id_create_index, sqlite_patient_index_query, sqlite_patient_create_query, \
    sqlite_encounter_create_index_bed_id_query, sqlite_encounter_create_index_patient_id_query, \
    sqlite_encounter_create_index_source_id_query, sqlite_encounter_create_query, sqlite_device_create_query, \
    sqlite_device_bed_id_create_index, sqlite_device_source_id_create_index, sqlite_insert_adb_source, \
    sqlite_measure_source_id_create_index, sqlite_log_hl7_adt_source_id_create_index, sqlite_log_hl7_adt_create_query, \
    sqlite_device_patient_table, sqlite_patient_table_index_1, sqlite_patient_history_create_query, \
    sqlite_encounter_insert_trigger, sqlite_encounter_update_trigger, sqlite_encounter_delete_trigger, \
    sqlite_label_set_create_query, sqlite_label_create_query, sqlite_label_source_create_query, \
    sqlite_label_table_index_1, sqlite_label_table_index_2, sqlite_patient_history_create_index_query


class SQLiteHandler(SQLHandler):
    _MISSING_COLUMN_PHRASE = "no such column"
    _MISSING_TABLE_ERRORS = (sqlite3.OperationalError,)
    _MISSING_TABLE_PHRASES = ("no such table",)

    def __init__(self, db_file: Union[str, Path]):
        self.db_file = db_file

    def sqlite_connect(self):
        conn = sqlite3.connect(self.db_file)

        return conn

    def connection(self, begin=False):
        return self.sqlite_db_connection(begin=begin)

    @contextmanager
    def sqlite_db_connection(self, begin=False):
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        try:
            if begin:
                cursor.execute("BEGIN")
            yield conn, cursor
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def create_schema(self):
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        # Create Tables
        cursor.execute(sqlite_source_create_query)
        cursor.execute(sqlite_institution_create_query)
        cursor.execute(sqlite_unit_create_query)

        cursor.execute(sqlite_bed_create_query)
        cursor.execute(sqlite_patient_create_query)
        cursor.execute(sqlite_patient_history_create_query)
        cursor.execute(sqlite_patient_history_create_index_query)

        cursor.execute(sqlite_encounter_create_query)

        cursor.execute(sqlite_measure_create_query)
        cursor.execute(sqlite_device_create_query)
        cursor.execute(sqlite_file_index_create_query)
        cursor.execute(sqlite_block_index_create_query)
        cursor.execute(sqlite_interval_index_create_query)
        cursor.execute(sqlite_settings_create_query)

        cursor.execute(sqlite_device_encounter_create_query)
        cursor.execute(sqlite_log_hl7_adt_create_query)
        cursor.execute(sqlite_device_patient_table)

        cursor.execute(sqlite_label_set_create_query)
        cursor.execute(sqlite_label_create_query)
        cursor.execute(sqlite_label_source_create_query)

        # Create Indices
        cursor.execute(sqlite_block_index_idx_query)
        cursor.execute(sqlite_interval_index_idx_query)
        cursor.execute(sqlite_unit_institution_id_create_index)

        cursor.execute(sqlite_bed_unit_id_create_index)
        cursor.execute(sqlite_patient_index_query)

        cursor.execute(sqlite_encounter_create_index_bed_id_query)
        cursor.execute(sqlite_encounter_create_index_patient_id_query)
        cursor.execute(sqlite_encounter_create_index_source_id_query)

        cursor.execute(sqlite_device_bed_id_create_index)
        cursor.execute(sqlite_device_source_id_create_index)

        cursor.execute(sqlite_measure_source_id_create_index)

        cursor.execute(sqlite_device_encounter_device_id_create_index)
        cursor.execute(sqlite_device_encounter_encounter_id_create_index)
        cursor.execute(sqlite_device_encounter_source_id_create_index)

        cursor.execute(sqlite_log_hl7_adt_source_id_create_index)
        cursor.execute(sqlite_patient_table_index_1)

        cursor.execute(sqlite_label_table_index_1)
        cursor.execute(sqlite_label_table_index_2)

        # Triggers
        cursor.execute(sqlite_encounter_insert_trigger)
        cursor.execute(sqlite_encounter_update_trigger)
        cursor.execute(sqlite_encounter_delete_trigger)

        # Insert Default Values
        cursor.execute(sqlite_insert_adb_source)

        conn.commit()
        cursor.close()
        conn.close()

        # Stamp the dataset a brand-new schema produces as current, so opening it
        # right after (even with the default auto_upgrade=False) never sees a
        # pending upgrade -- see SQLHandler.pending_schema_upgrades.
        self.record_dataset_schema_version()

    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns

    def update_measure_schema(self):
        """Add the additive nullable measure columns if they do not exist.

        Idempotent, additive migration (safe on write-once production history):
        ``period_ns`` plus the measure-kind columns ``signal_kind``
        and ``value_type``. A ``NULL`` in any of these is interpreted with a
        read-time default by the SDK (``signal_kind`` -> ``waveform``,
        ``value_type`` -> ``numeric``), so existing rows need no backfill for
        correctness. Returns True if any column was added."""
        changed = False
        with self.connection() as (conn, cursor):
            if not self._column_exists(cursor, 'measure', 'period_ns'):
                cursor.execute("ALTER TABLE measure ADD COLUMN period_ns INTEGER NULL")
                changed = True
            if not self._column_exists(cursor, 'measure', 'signal_kind'):
                cursor.execute("ALTER TABLE measure ADD COLUMN signal_kind TEXT NULL")
                changed = True
            if not self._column_exists(cursor, 'measure', 'value_type'):
                cursor.execute("ALTER TABLE measure ADD COLUMN value_type TEXT NULL")
                changed = True
            if changed:
                conn.commit()
        return changed

    def check_mrn_column_is_text(self) -> bool:
        """Check if the mrn column in the patient table is TEXT. Returns True if it is TEXT."""
        with self.connection() as (conn, cursor):
            cursor.execute("PRAGMA table_info(patient)")
            for row in cursor.fetchall():
                # row format: (cid, name, type, notnull, dflt_value, pk)
                if row[1] == 'mrn':
                    return row[2].upper() == 'TEXT'
        return False

    def upgrade_mrn_schema(self):
        """Upgrade the patient table mrn column from INTEGER to TEXT if needed.

        SQLite does not support ALTER COLUMN, so we must recreate the table.
        """
        if self.check_mrn_column_is_text():
            return False  # Already TEXT, no upgrade needed

        with self.connection() as (conn, cursor):
            cursor.execute("PRAGMA foreign_keys = OFF")

            # Create a new patient table with TEXT mrn
            cursor.execute("""
                CREATE TABLE patient_new (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  mrn TEXT NULL UNIQUE,
                  gender TEXT NULL,
                  dob INTEGER NULL,
                  first_name TEXT NULL,
                  middle_name TEXT NULL,
                  last_name TEXT NULL,
                  first_seen INTEGER NULL DEFAULT (STRFTIME('%s','NOW')),
                  last_updated INTEGER NULL,
                  source_id INTEGER DEFAULT 1 NULL,
                  weight REAL NULL,
                  height REAL NULL,
                  FOREIGN KEY (source_id) REFERENCES source (id)
                )
            """)

            # Copy data, casting mrn to TEXT
            cursor.execute("""
                INSERT INTO patient_new (id, mrn, gender, dob, first_name, middle_name, last_name,
                    first_seen, last_updated, source_id, weight, height)
                SELECT id, CAST(mrn AS TEXT), gender, dob, first_name, middle_name, last_name,
                    first_seen, last_updated, source_id, weight, height
                FROM patient
            """)

            # Drop old table and rename new one
            cursor.execute("DROP TABLE patient")
            cursor.execute("ALTER TABLE patient_new RENAME TO patient")

            # Recreate the index
            cursor.execute("CREATE INDEX IF NOT EXISTS source_id ON patient (source_id)")

            cursor.execute("PRAGMA foreign_keys = ON")
            conn.commit()
            return True

    def interval_exists(self, measure_id, device_id, start_time_nano):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_interval_exists_query, (measure_id, device_id, start_time_nano))
            result = cursor.fetchone()
        return result[0]

    def select_all_measures(self):
        try:
            with self.sqlite_db_connection() as (conn, cursor):
                cursor.execute("SELECT id, tag, name, freq_nhz, period_ns, code, unit, unit_label, unit_code, source_id, signal_kind, value_type FROM measure")
                rows = cursor.fetchall()
            return rows
        except sqlite3.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def select_all_patients(self):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute("SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient")
            rows = cursor.fetchall()
        return rows

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None,
                       measure_id=None, code: str = None, unit_label: str = None, unit_code: str = None,
                       source_id: int = None, period_ns: int = None, signal_kind: str = None,
                       value_type: str = None):
        units = "" if units is None else units

        try:
            with self.connection() as (conn, cursor):
                cursor.execute(
                    "INSERT OR IGNORE INTO measure (id, tag, freq_nhz, period_ns, unit, name, code, unit_label, unit_code, source_id, signal_kind, value_type) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (measure_id, measure_tag, freq_nhz, period_ns, units, measure_name, code, unit_label, unit_code,
                     source_id, signal_kind, value_type))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units

        try:
            with self.sqlite_db_connection() as (conn, cursor):
                if measure_id is not None:
                    cursor.execute(sqlite_select_measure_from_id_query, (measure_id,))
                else:
                    cursor.execute(sqlite_select_measure_from_triplet_query,
                                   (measure_tag, freq_nhz, units))
                row = cursor.fetchone()
            return row
        except sqlite3.Error as e:
            self._reraise_missing_measure_column(e)
            raise

    def set_string_dict_watermark(self, measure_id: int, vocabulary_size: int):
        """Raise this measure's recorded string-vocabulary size (see
        ``SQLHandler.set_string_dict_watermark``). One statement, so concurrent
        writers cannot interleave a read and a write and lower the mark; the
        ``WHERE`` clause makes it monotonic."""
        name = self._string_dict_size_setting_name(measure_id)
        value = str(int(vocabulary_size))
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            cursor.execute(
                "INSERT INTO setting (name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value "
                "WHERE CAST(setting.value AS INTEGER) < CAST(excluded.value AS INTEGER)",
                (name, value))

    def _upsert_setting(self, name: str, value: str):
        """Unconditional single-statement upsert for one `setting` row (see
        ``SQLHandler._upsert_setting``) -- unlike ``set_string_dict_watermark``
        above, there is no monotonic comparison; the new value always wins."""
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            cursor.execute(
                "INSERT INTO setting (name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                (name, value))

    def insert_device(self, device_tag: str, device_name: str = None, device_id=None, manufacturer: str = None,
                      model: str = None, device_type: str = None, bed_id: int = None, source_id: int = None):
        device_type = "static" if device_type is None else device_type
        source_id = 1 if source_id is None else source_id
        with self.connection() as (conn, cursor):
            cursor.execute(
                "INSERT OR IGNORE INTO device (id, tag, name, manufacturer, model, type, bed_id, source_id) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?);",
                (device_id, device_tag, device_name, manufacturer, model, device_type, bed_id, source_id))
            conn.commit()

            return cursor.lastrowid

    def _insert_intervals(self, cursor, interval_data: List[Dict], interval_index_mode, gap_tolerance: int = 0):
        """Insert interval rows using the given interval_index_mode. "fast" appends
        raw rows; "merge" unions each new interval with every existing row within
        gap_tolerance so the index stays sparse; "disable" does nothing."""
        if interval_index_mode == "disable":
            return

        if interval_index_mode == "merge":
            for interval in interval_data:
                measure_id, device_id = interval["measure_id"], interval["device_id"]
                start, end = int(interval["start_time_n"]), int(interval["end_time_n"])

                # Find every existing interval within gap_tolerance of the new one.
                cursor.execute(
                    "SELECT id, start_time_n, end_time_n FROM interval_index "
                    "WHERE measure_id = ? AND device_id = ? AND start_time_n <= ? AND end_time_n >= ?",
                    (measure_id, device_id, end + gap_tolerance, start - gap_tolerance))
                rows = cursor.fetchall()

                if rows:
                    # Union them all (the new interval may bridge several rows) into
                    # a single row, reusing the first and deleting the rest.
                    start = min(start, min(row[1] for row in rows))
                    end = max(end, max(row[2] for row in rows))
                    if len(rows) > 1:
                        cursor.executemany("DELETE FROM interval_index WHERE id = ?",
                                           [(row[0],) for row in rows[1:]])
                    cursor.execute("UPDATE interval_index SET start_time_n = ?, end_time_n = ? WHERE id = ?",
                                   (start, end, rows[0][0]))
                else:
                    cursor.execute(sqlite_insert_interval_index_query, (measure_id, device_id, start, end))
            return

        # "fast" (and legacy None): append raw rows.
        cursor.executemany(sqlite_insert_interval_index_query, interval_insert_tuples(interval_data))

    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode, gap_tolerance: int = 0):
        # default to fast
        interval_index_mode = "fast" if interval_index_mode is None else interval_index_mode

        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(sqlite_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            cursor.executemany(sqlite_insert_block_query, block_insert_tuples(block_data, file_id))

            # insert into interval_index
            self._insert_intervals(cursor, interval_data, interval_index_mode, gap_tolerance)

    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int], gap_tolerance: int = 0):
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            # insert/update file data
            for file_path, (block_data, interval_data) in file_data.items():
                # insert file_path into file_index and get id
                cursor.execute(sqlite_insert_file_index_query, (file_path,))
                file_id = cursor.lastrowid

                # insert into block_index
                cursor.executemany(sqlite_insert_block_query, block_insert_tuples(block_data, file_id))

                # insert into interval_index
                cursor.executemany(sqlite_insert_interval_index_query, interval_insert_tuples(interval_data))

            # delete old block data (Don't need, triggered automatically)
            cursor.executemany(sqlite_delete_block_query, [(block_id,) for block_id in block_ids_to_delete])

            # delete old file data
            # cursor.executemany(sqlite_delete_file_query, [(file_id,) for file_id in file_ids_to_delete])

    def insert_merged_block_data(self, file_path: str, block_data: List[Dict], old_block: tuple, interval_data: List[Dict],
                                 interval_index_mode, gap_tolerance: int = 0):
        # default to fast
        interval_index_mode = "fast" if interval_index_mode is None else interval_index_mode

        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(sqlite_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            cursor.executemany(sqlite_insert_block_query, block_insert_tuples(block_data, file_id))

            # insert into interval_index
            self._insert_intervals(cursor, interval_data, interval_index_mode, gap_tolerance)

            return self._delete_merged_block(cursor, old_block)

    def select_block(self, block_id: int = None, measure_id: int = None, device_id: int = None, file_id: int = None,
                     start_byte: int = None, num_bytes: int = None, start_time_n: int = None, end_time_n: int = None,
                     num_values: int = None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            if block_id is not None:
                cursor.execute(sqlite_select_block_by_id, (block_id,))
            else:
                cursor.execute(sqlite_select_block_by_values, (measure_id, device_id, file_id, start_byte,
                                                               num_bytes, start_time_n, end_time_n, num_values))
            row = cursor.fetchone()
        return row

    def insert_setting(self, setting_name: str, setting_value: str):
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            cursor.execute(sqlite_setting_insert_query, (setting_name, setting_value))
            conn.commit()

    def insert_patient(self, patient_id=None, mrn=None, gender=None, dob=None, first_name=None, middle_name=None,
                       last_name=None, first_seen=None, last_updated=None, source_id=1, weight=None, height=None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_patient_query,
                           (patient_id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated,
                            source_id, weight, height))
            return cursor.lastrowid

    def insert_encounter(self, patient_id, bed_id, start_time, end_time=None, source_id=1, visit_number=None,
                         last_updated=None):
        last_updated = time.time_ns() if last_updated is None else last_updated
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_encounter_query,
                           (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated))
            return cursor.lastrowid

    def insert_device_encounter(self, device_id, encounter_id, start_time, end_time=None, source_id=1):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_device_encounter_query,
                           (device_id, encounter_id, start_time, end_time, source_id))
            return cursor.lastrowid

    def select_encounters(self, patient_id_list: List[int] = None, mrn_list: List[str] = None, start_time: int = None,
                          end_time: int = None):
        assert (patient_id_list is None) != (
                mrn_list is None), "Either patient_id_list or mrn_list must be provided, but not both"
        # An empty filter list names no patients, so no encounters. Building
        # `IN ()` instead is a syntax error on MariaDB.
        if patient_id_list is not None and len(patient_id_list) == 0:
            return []
        if mrn_list is not None and len(mrn_list) == 0:
            return []

        arg_tuple = ()
        sqlite_select_encounter_query = \
            "SELECT encounter.id, encounter.patient_id, encounter.bed_id, encounter.start_time, encounter.end_time, " \
            "encounter.source_id, encounter.visit_number, encounter.last_updated FROM encounter"
        if patient_id_list is not None:
            sqlite_select_encounter_query += \
                " INNER JOIN patient ON encounter.patient_id = patient.id WHERE encounter.patient_id IN ({})".format(
                    ','.join(['?'] * len(patient_id_list)))
            arg_tuple += tuple(patient_id_list)
        else:
            sqlite_select_encounter_query += \
                " INNER JOIN patient ON encounter.patient_id = patient.id WHERE patient.mrn IN ({})".format(
                    ','.join(['?'] * len(mrn_list)))
            arg_tuple += tuple(mrn_list)
        if start_time is not None:
            sqlite_select_encounter_query += " AND encounter.end_time > ?"
            arg_tuple += (start_time,)
        if end_time is not None:
            sqlite_select_encounter_query += " AND encounter.start_time < ?"
            arg_tuple += (end_time,)
        sqlite_select_encounter_query += " ORDER BY encounter.id ASC"

        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_select_encounter_query, arg_tuple)
            return cursor.fetchall()

    def select_all_patients_in_list(self, patient_id_list: List[int] = None, mrn_list: List[str] = None):
        if patient_id_list is not None:
            # An empty filter list names no rows. Building `IN ()` instead is a
            # syntax error on MariaDB and a silent "matches nothing" on SQLite.
            if len(patient_id_list) == 0:
                return []
            placeholders = ', '.join(['?'] * len(patient_id_list))
            sqlite_select_patients_by_id_list = f"SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient WHERE id IN ({placeholders})"
        elif mrn_list is not None:
            # An empty filter list names no rows. Building `IN ()` instead is a
            # syntax error on MariaDB and a silent "matches nothing" on SQLite.
            if len(mrn_list) == 0:
                return []
            patient_id_list = mrn_list
            placeholders = ', '.join(['?'] * len(patient_id_list))
            sqlite_select_patients_by_id_list = f"SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient WHERE mrn IN ({placeholders})"
        else:
            sqlite_select_patients_by_id_list = "SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient"
            patient_id_list = tuple()

        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_patients_by_id_list, patient_id_list)
            rows = cursor.fetchall()
        return rows

