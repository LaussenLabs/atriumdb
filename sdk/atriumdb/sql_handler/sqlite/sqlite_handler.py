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
from atriumdb.sql_handler.sql_handler import SQLHandler
from atriumdb.sql_handler.sql_helper import join_sql_and_bools
from atriumdb.sql_handler.sqlite.sqlite_functions import sqlite_insert_ignore_measure_query, \
    sqlite_select_measure_from_triplet_query, sqlite_select_measure_from_id_query, sqlite_insert_ignore_device_query, \
    sqlite_select_device_from_tag_query, sqlite_select_device_from_id_query, sqlite_insert_file_index_query, \
    sqlite_insert_block_query, sqlite_insert_interval_index_query, sqlite_select_file_by_id, \
    sqlite_select_file_by_values, sqlite_select_block_by_id, sqlite_select_block_by_values, \
    sqlite_select_interval_by_id, sqlite_select_interval_by_values, sqlite_setting_select_query, \
    sqlite_setting_insert_query, sqlite_setting_select_all_query, sqlite_insert_ignore_source_query, \
    sqlite_insert_ignore_institution_query, sqlite_insert_ignore_unit_query, sqlite_insert_ignore_bed_query, \
    sqlite_insert_ignore_patient_query, sqlite_insert_ignore_encounter_query, \
    sqlite_insert_ignore_device_encounter_query, sqlite_select_blocks_from_file, sqlite_delete_block_query, \
    sqlite_delete_file_query, sqlite_interval_exists_query, sqlite_get_query_with_patient_id
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
    sqlite_encounter_insert_trigger, sqlite_encounter_update_trigger, sqlite_encounter_delete_trigger


class SQLiteHandler(SQLHandler):
    def __init__(self, db_file: Union[str, Path]):
        self.db_file = db_file

    def sqlite_connect(self):
        conn = sqlite3.connect(self.db_file)

        return conn

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

        # Triggers
        cursor.execute(sqlite_encounter_insert_trigger)
        cursor.execute(sqlite_encounter_update_trigger)
        cursor.execute(sqlite_encounter_delete_trigger)

        # Insert Default Values
        cursor.execute(sqlite_insert_adb_source)

        conn.commit()
        cursor.close()
        conn.close()

    def interval_exists(self, measure_id, device_id, start_time_nano):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_interval_exists_query, (measure_id, device_id, start_time_nano))
            result = cursor.fetchone()
        return result[0]

    def select_all_devices(self):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute("SELECT id, tag, name, manufacturer, model, type, bed_id, source_id FROM device")
            rows = cursor.fetchall()
        return rows

    def select_all_measures(self):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute("SELECT id, tag, name, freq_nhz, code, unit, unit_label, unit_code, source_id FROM measure")
            rows = cursor.fetchall()
        return rows

    def select_all_patients(self):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute("SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient")
            rows = cursor.fetchall()
        return rows

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None):
        units = DEFAULT_UNITS if units is None else units

        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_measure_query, (measure_tag, freq_nhz, units, measure_name))
            conn.commit()
            cursor.execute(sqlite_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
            measure_id = cursor.fetchone()[0]
        return measure_id

    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units

        with self.sqlite_db_connection() as (conn, cursor):
            if measure_id is not None:
                cursor.execute(sqlite_select_measure_from_id_query, (measure_id,))
            else:
                cursor.execute(sqlite_select_measure_from_triplet_query,
                               (measure_tag, freq_nhz, units))
            row = cursor.fetchone()
        return row

    def insert_device(self, device_tag: str, device_name: str = None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_device_query, (device_tag, device_name))
            conn.commit()
            cursor.execute(sqlite_select_device_from_tag_query, (device_tag,))
            device_id = cursor.fetchone()[0]
        return device_id

    def select_device(self, device_id: int = None, device_tag: str = None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            if device_id is not None:
                cursor.execute(sqlite_select_device_from_id_query, (device_id,))
            else:
                cursor.execute(sqlite_select_device_from_tag_query, (device_tag,))
            row = cursor.fetchone()
        return row

    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode):
        # default to merge
        interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode

        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(sqlite_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            block_tuples = [(block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
                             block["start_time_n"], block["end_time_n"], block["num_values"])
                            for block in block_data]
            cursor.executemany(sqlite_insert_block_query, block_tuples)

            # insert into interval_index
            if interval_index_mode != "disable":
                interval_tuples = [(interval["measure_id"], interval["device_id"], interval["start_time_n"],
                                    interval["end_time_n"]) for interval in interval_data]
                cursor.executemany(sqlite_insert_interval_index_query, interval_tuples)

    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int]):
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            # insert/update file data
            for file_path, (block_data, interval_data) in file_data.items():
                # insert file_path into file_index and get id
                cursor.execute(sqlite_insert_file_index_query, (file_path,))
                file_id = cursor.lastrowid

                # insert into block_index
                block_tuples = [
                    (block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
                     block["start_time_n"], block["end_time_n"], block["num_values"])
                    for block in block_data]
                cursor.executemany(sqlite_insert_block_query, block_tuples)

                # insert into interval_index
                interval_tuples = [(interval["measure_id"], interval["device_id"], interval["start_time_n"],
                                    interval["end_time_n"]) for interval in interval_data]
                cursor.executemany(sqlite_insert_interval_index_query, interval_tuples)

            # delete old block data (Don't need, triggered automatically)
            cursor.executemany(sqlite_delete_block_query, [(block_id,) for block_id in block_ids_to_delete])

            # delete old file data
            # cursor.executemany(sqlite_delete_file_query, [(file_id,) for file_id in file_ids_to_delete])

    def select_file(self, file_id: int = None, file_path: str = None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            if file_id is not None:
                cursor.execute(sqlite_select_file_by_id, (file_id,))
            else:
                cursor.execute(sqlite_select_file_by_values, (file_path,))
            row = cursor.fetchone()
        return row

    def select_files(self, file_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(file_id_list))
        sqlite_select_files_by_id_list = f"SELECT id, path FROM file_index WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_files_by_id_list, file_id_list)
            rows = cursor.fetchall()
        return rows

    def select_blocks_from_file(self, file_id: int):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_blocks_from_file, (file_id,))
            rows = cursor.fetchall()
        return rows

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

    def select_interval(self, interval_id: int = None, measure_id: int = None, device_id: int = None,
                        start_time_n: int = None, end_time_n: int = None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            if interval_id is not None:
                cursor.execute(sqlite_select_interval_by_id, (interval_id,))
            else:
                cursor.execute(sqlite_select_interval_by_values, (measure_id, device_id, start_time_n, end_time_n))
            row = cursor.fetchone()
        return row

    def insert_setting(self, setting_name: str, setting_value: str):
        with self.sqlite_db_connection(begin=True) as (conn, cursor):
            cursor.execute(sqlite_setting_insert_query, (setting_name, setting_value))
            conn.commit()

    def select_setting(self, setting_name: str):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_setting_select_query, (setting_name,))
            setting = cursor.fetchone()
        return setting

    def select_all_settings(self):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_setting_select_all_query)
            settings = cursor.fetchall()
        return settings

    def insert_source(self, s_name, description=None):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_source_query, (s_name, description))
            return cursor.lastrowid

    def insert_institution(self, i_name):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_institution_query, (i_name,))
            return cursor.lastrowid

    def insert_unit(self, institution_id, u_name, u_type):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_unit_query, (institution_id, u_name, u_type))
            return cursor.lastrowid

    def insert_bed(self, unit_id, b_name):
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_bed_query, (unit_id, b_name))
            return cursor.lastrowid

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

    def select_blocks(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        assert device_id is not None or patient_id is not None, "Either device_id or patient_id must be provided"

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

            block_query = """
                            SELECT
                                id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
                            FROM
                                block_index
                            WHERE
                                measure_id = ? AND device_id = ? AND end_time_n >= ? AND start_time_n <= ?
                            ORDER BY
                                file_id, start_byte ASC"""

            block_results = []

            with self.sqlite_db_connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    args = (measure_id, encounter_device_id, encounter_start_time, encounter_end_time)

                    cursor.execute(block_query, args)
                    block_results.extend(cursor.fetchall())

            return block_results

        # Query by device.
        block_query = """
                        SELECT
                            id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
                        FROM
                            block_index
                        WHERE
                            measure_id = ? AND device_id = ?"""

        args = (measure_id, device_id)

        if end_time_n is not None:
            block_query += " AND start_time_n <= ?"
            args += (end_time_n,)

        if start_time_n is not None:
            block_query += " AND end_time_n >= ?"
            args += (start_time_n,)

        block_query += " ORDER BY file_id, start_byte ASC"

        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def get_device_time_ranges_by_patient(self, patient_id, end_time_n, start_time_n):
        patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ?"
        args = (patient_id,)
        if start_time_n is not None:
            patient_device_query += " AND end_time >= ? "
            args += (start_time_n,)
        if end_time_n is not None:
            patient_device_query += " AND start_time <= ? "
            args += (end_time_n,)
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            args = (patient_id,) + ((start_time_n,) if start_time_n is not None else ()) + (
                (end_time_n,) if end_time_n is not None else ())
            cursor.execute(patient_device_query, args)
            return cursor.fetchall()

    # def select_intervals(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
    #     assert device_id is not None or patient_id is not None, "Either device_id or patient_id must be provided"
    #
    #     if patient_id is not None:
    #         device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, start_time_n, end_time_n)
    #     else:
    #         device_time_ranges = [(device_id, start_time_n, end_time_n)]
    #
    #     interval_query = """
    #                     SELECT
    #                         id, measure_id, device_id, start_time_n, end_time_n
    #                     FROM
    #                         interval_index
    #                     WHERE
    #                         measure_id = ? AND device_id = ? AND end_time_n >= ? AND start_time_n <= ?"""
    #
    #     block_results = []
    #
    #     with self.sqlite_db_connection(begin=False) as (conn, cursor):
    #         for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
    #             args = (measure_id, encounter_device_id, encounter_start_time, encounter_end_time)
    #
    #             cursor.execute(interval_query, args)
    #             block_results.extend(cursor.fetchall())
    #
    #     return block_results

    def select_intervals(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        assert device_id is not None or patient_id is not None, "Either device_id or patient_id must be provided"

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

            interval_query = """
                                    SELECT
                                        id, measure_id, device_id, start_time_n, end_time_n
                                    FROM
                                        interval_index
                                    WHERE
                                        measure_id = ? AND device_id = ? AND end_time_n >= ? AND start_time_n <= ?"""

            interval_results = []

            with self.sqlite_db_connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    args = (measure_id, encounter_device_id, encounter_start_time, encounter_end_time)

                    cursor.execute(interval_query, args)
                    interval_results.extend(cursor.fetchall())

            return interval_results

        # Query by device.

        interval_query = """
                        SELECT
                            id, measure_id, device_id, start_time_n, end_time_n
                        FROM
                            interval_index
                        WHERE
                            measure_id = ? AND device_id = ?"""

        args = (measure_id, device_id)

        if end_time_n is not None:
            interval_query += " AND start_time_n <= ?"
            args += (end_time_n,)

        if start_time_n is not None:
            interval_query += " AND end_time_n >= ?"
            args += (start_time_n,)

        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(interval_query, args)
            return cursor.fetchall()

    def select_encounters(self, patient_id_list: List[int] = None, mrn_list: List[int] = None, start_time: int = None,
                          end_time: int = None):
        assert (patient_id_list is None) != (
                mrn_list is None), "Either patient_id_list or mrn_list must be provided, but not both"
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

    def select_all_measures_in_list(self, measure_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(measure_id_list))
        sqlite_select_measures_by_id_list = f"SELECT id, tag, name, freq_nhz, code, unit, unit_label, unit_code, source_id FROM measure WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_measures_by_id_list, measure_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_patients_in_list(self, patient_id_list: List[int] = None, mrn_list: List[int] = None):
        assert (patient_id_list is None) != (mrn_list is None), \
            "only one of patient_id_list and mrn_list can be specified."
        if patient_id_list is not None:
            placeholders = ', '.join(['?'] * len(patient_id_list))
            sqlite_select_patients_by_id_list = f"SELECT id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height FROM patient WHERE id IN ({placeholders})"
        elif mrn_list is not None:
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

    def select_all_devices_in_list(self, device_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(device_id_list))
        sqlite_select_devices_by_id_list = f"SELECT id, tag, name, manufacturer, model, type, bed_id, source_id FROM device WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_devices_by_id_list, device_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_beds_in_list(self, bed_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(bed_id_list))
        sqlite_select_beds_by_id_list = f"SELECT id, unit_id, name FROM bed WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_beds_by_id_list, bed_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_units_in_list(self, unit_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(unit_id_list))
        sqlite_select_units_by_id_list = f"SELECT id, institution_id, name, type FROM unit WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_units_by_id_list, unit_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_institutions_in_list(self, institution_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(institution_id_list))
        sqlite_select_institutions_by_id_list = f"SELECT id, name FROM institution WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_institutions_by_id_list, institution_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_device_encounters_by_encounter_list(self, encounter_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(encounter_id_list))
        sqlite_select_device_encounters_by_encounter_list = f"SELECT id, device_id, encounter_id, start_time, end_time, source_id FROM device_encounter WHERE encounter_id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_device_encounters_by_encounter_list, encounter_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_sources_in_list(self, source_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(source_id_list))
        sqlite_select_sources_by_id_list = f"SELECT id, name, description FROM source WHERE id IN ({placeholders})"
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_sources_by_id_list, source_id_list)
            rows = cursor.fetchall()
        return rows

    def select_device_patients(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                               start_time: int = None, end_time: int = None):
        arg_tuple = ()
        sqlite_select_device_patient_query = \
            "SELECT device_id, patient_id, start_time, end_time FROM device_patient"
        where_clauses = []
        if device_id_list is not None and len(device_id_list) > 0:
            where_clauses.append("device_id IN ({})".format(
                ','.join(['?'] * len(device_id_list))))
            arg_tuple += tuple(device_id_list)
        if patient_id_list is not None and len(patient_id_list) > 0:
            where_clauses.append("patient_id IN ({})".format(
                ','.join(['?'] * len(patient_id_list))))
            arg_tuple += tuple(patient_id_list)
        if start_time is not None:
            where_clauses.append("end_time > ?")
            arg_tuple += (start_time,)
        if end_time is not None:
            where_clauses.append("start_time < ?")
            arg_tuple += (end_time,)
        sqlite_select_device_patient_query += join_sql_and_bools(where_clauses)
        sqlite_select_device_patient_query += " ORDER BY id ASC"

        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute(sqlite_select_device_patient_query, arg_tuple)
            return cursor.fetchall()

    def insert_device_patients(self, device_patient_data: List[Tuple[int, int, int, int]]):
        sqlite_insert_device_patient_query = \
            "INSERT INTO device_patient (device_id, patient_id, start_time, end_time) VALUES (?, ?, ?, ?)"

        with self.sqlite_db_connection() as (conn, cursor):
            cursor.executemany(sqlite_insert_device_patient_query, device_patient_data)
            conn.commit()
