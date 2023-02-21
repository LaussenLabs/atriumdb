import sqlite3
from pathlib import Path
from typing import Union, List, Dict, Tuple
from contextlib import contextmanager
import time

from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler
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
    sqlite_delete_file_query, sqlite_interval_exists_query
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
    sqlite_measure_source_id_create_index, sqlite_log_hl7_adt_source_id_create_index, sqlite_log_hl7_adt_create_query


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

        cursor.execute(sqlite_encounter_create_query)

        cursor.execute(sqlite_measure_create_query)
        cursor.execute(sqlite_device_create_query)
        cursor.execute(sqlite_file_index_create_query)
        cursor.execute(sqlite_block_index_create_query)
        cursor.execute(sqlite_interval_index_create_query)
        cursor.execute(sqlite_settings_create_query)

        cursor.execute(sqlite_device_encounter_create_query)
        cursor.execute(sqlite_log_hl7_adt_create_query)

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
            cursor.execute("SELECT * FROM device")
            rows = cursor.fetchall()
        return rows

    def select_all_measures(self):
        with self.sqlite_db_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM measure")
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

    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict]):
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

            # delete old block data
            cursor.executemany(sqlite_delete_block_query, [(block_id,) for block_id in block_ids_to_delete])

            # delete old file data
            cursor.executemany(sqlite_delete_file_query, [(file_id,) for file_id in file_ids_to_delete])

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
        sqlite_select_files_by_id_list = f"SELECT * FROM file_index WHERE id IN ({placeholders})"
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

    def insert_patient(self, mrn, gender=None, dob=None, first_name=None, middle_name=None, last_name=None,
                       first_seen=None,
                       last_updated=None, source_id=1):
        first_seen = time.time_ns() if first_seen is None else first_seen
        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_insert_ignore_patient_query,
                           (mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id))
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
        """
        block_index.measure_id = measure_id
        if start_time_n is not Null: block_index.end_time_n > start_time_n
        if end_time_n is not Null: block_index.start_time_n < end_time_n
        if device_id is not Null: block_index.device_id = device_id
        if patient_id is not Null:
            encounter.patient_id = patient_id
            and encounter.id = device_encounter.encounter_id
            and device_encounter.device_id = block_index.device_id
            and device_encounter.start_time < block_index.end_time_n
            and block_index.start_time < device_encounter.end_time_n
        """
        arg_tuple = ()
        sqlite_select_block_query = "SELECT block_index.id, block_index.measure_id, block_index.device_id, block_index.file_id, block_index.start_byte, block_index.num_bytes, block_index.start_time_n, block_index.end_time_n, block_index.num_values FROM block_index"
        if patient_id is not None:
            sqlite_select_block_query += " INNER JOIN device_encounter ON device_encounter.device_id = block_index.device_id"
            sqlite_select_block_query += " INNER JOIN encounter ON encounter.id = device_encounter.encounter_id"
        sqlite_select_block_query += " WHERE block_index.measure_id = ?"
        arg_tuple += (measure_id,)
        if start_time_n is not None:
            sqlite_select_block_query += " AND block_index.end_time_n > ?"
            arg_tuple += (start_time_n,)
        if end_time_n is not None:
            sqlite_select_block_query += " AND block_index.start_time_n < ?"
            arg_tuple += (end_time_n,)
        if device_id is not None:
            sqlite_select_block_query += " AND block_index.device_id = ?"
            arg_tuple += (device_id,)
        if patient_id is not None:
            sqlite_select_block_query += " AND encounter.patient_id = ?"
            arg_tuple += (patient_id,)
            sqlite_select_block_query += " AND device_encounter.start_time < block_index.end_time_n"
            sqlite_select_block_query += " AND block_index.start_time_n < device_encounter.end_time"
        sqlite_select_block_query += " ORDER BY block_index.file_id ASC, block_index.start_byte ASC"

        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_select_block_query, arg_tuple)
            return cursor.fetchall()

    def select_intervals(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        """
        interval_index.measure_id = measure_id
        if start_time_n is not Null: interval_index.end_time_n > start_time_n
        if end_time_n is not Null: interval_index.start_time_n < end_time_n
        if device_id is not Null: interval_index.device_id = device_id
        if patient_id is not Null:
            encounter.patient_id = patient_id
            and encounter.id = device_encounter.encounter_id
            and device_encounter.device_id = interval_index.device_id
            and device_encounter.start_time < interval_index.end_time_n
            and interval_index.start_time < device_encounter.end_time_n
        """
        arg_tuple = ()
        sqlite_select_interval_query = "SELECT interval_index.id, interval_index.measure_id, interval_index.device_id, interval_index.start_time_n, interval_index.end_time_n FROM interval_index"
        if patient_id is not None:
            sqlite_select_interval_query += " INNER JOIN device_encounter ON device_encounter.device_id = interval_index.device_id"
            sqlite_select_interval_query += " INNER JOIN encounter ON encounter.id = device_encounter.encounter_id"
        sqlite_select_interval_query += " WHERE interval_index.measure_id = ?"
        arg_tuple += (measure_id,)
        if start_time_n is not None:
            sqlite_select_interval_query += " AND interval_index.end_time_n > ?"
            arg_tuple += (start_time_n,)
        if end_time_n is not None:
            sqlite_select_interval_query += " AND interval_index.start_time_n < ?"
            arg_tuple += (end_time_n,)
        if device_id is not None:
            sqlite_select_interval_query += " AND interval_index.device_id = ?"
            arg_tuple += (device_id,)
        if patient_id is not None:
            sqlite_select_interval_query += " AND encounter.patient_id = ?"
            arg_tuple += (patient_id,)
            sqlite_select_interval_query += " AND device_encounter.start_time < interval_index.end_time_n"
            sqlite_select_interval_query += " AND interval_index.start_time_n < device_encounter.end_time"
        sqlite_select_interval_query += " ORDER BY interval_index.start_time_n ASC"

        with self.sqlite_db_connection(begin=False) as (conn, cursor):
            cursor.execute(sqlite_select_interval_query, arg_tuple)
            return cursor.fetchall()
