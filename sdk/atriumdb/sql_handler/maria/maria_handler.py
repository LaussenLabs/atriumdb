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
from typing import List, Dict, Tuple
from contextlib import contextmanager

import mariadb
import uuid

from atriumdb.sql_handler.maria.maria_functions import maria_select_measure_from_triplet_query, \
    maria_select_measure_from_id, \
    maria_insert_ignore_measure_query, maria_insert_ignore_device_query, maria_select_device_from_tag_query, \
    maria_select_device_from_id_query, maria_insert_file_index_query, maria_insert_block_query, \
    maria_select_file_by_id, maria_select_file_by_values, maria_select_block_by_id, \
    maria_select_block_by_values, maria_select_interval_by_id, maria_select_interval_by_values, \
    mariadb_setting_insert_query, mariadb_setting_select_query, maria_select_all_query, \
    maria_insert_ignore_device_encounter_query, maria_insert_ignore_encounter_query, maria_insert_ignore_patient_query, \
    maria_insert_ignore_bed_query, maria_insert_ignore_unit_query, maria_insert_ignore_institution_query, \
    maria_insert_ignore_source_query, maria_select_blocks_from_file, maria_delete_block_query, maria_delete_file_query, \
    maria_interval_exists_query
from atriumdb.sql_handler.maria.maria_tables import mariadb_measure_create_query, \
    maria_file_index_create_query, maria_block_index_create_query, maria_interval_index_create_query, \
    maria_settings_create_query, maria_device_encounter_create_query, maria_source_create_query, \
    maria_institution_create_query, maria_unit_create_query, maria_bed_create_query, maria_patient_create_query,\
    maria_encounter_create_query, mariadb_device_create_query, maria_insert_adb_source, \
    mariadb_log_hl7_adt_create_query, mariadb_current_census_view, mariadb_device_patient_table, \
    maria_encounter_device_encounter_insert_trigger, maria_encounter_device_encounter_update_trigger, \
    maria_encounter_device_patient_insert_trigger, maria_encounter_device_patient_update_trigger, \
    maria_insert_interval_stored_procedure
from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler
from atriumdb.sql_handler.sql_helper import join_sql_and_bools
import logging

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 3306


class MariaDBHandler(SQLHandler):
    def __init__(self, host: str, user: str, password: str, database: str, port: int = None, pool_size: int = 1,
                 no_pool=False):
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
        self.pool_size = pool_size
        self.no_pool = no_pool
        self.pool = None

    def maria_connect(self):
        return mariadb.connect(**self.connection_params)

    def create_pool(self):
        pool_guid = str(uuid.uuid4())
        self.pool = mariadb.ConnectionPool(pool_name=f"atriumdb_pool_{pool_guid}", pool_size=self.pool_size,
                                           **self.connection_params)

    def get_connection_from_pool(self):
        if self.pool is None:
            self.create_pool()
        return self.pool.get_connection()

    def maria_connect_no_db(self, db_name=None):
        conn = mariadb.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=db_name)

        return conn

    @contextmanager
    def maria_db_connection(self, begin=False):
        if self.no_pool:
            conn = self.maria_connect()
        else:
            conn = self.get_connection_from_pool()
        cursor = conn.cursor()

        try:
            if begin:
                cursor.execute("START TRANSACTION")
            yield conn, cursor
            conn.commit()
        except mariadb.Error as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

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

        # Create Views
        cursor.execute(mariadb_current_census_view)

        # Insert Default Values
        cursor.execute(maria_insert_adb_source)

        # Triggers
        cursor.execute(maria_encounter_device_encounter_insert_trigger)
        cursor.execute(maria_encounter_device_encounter_update_trigger)
        cursor.execute(maria_encounter_device_patient_insert_trigger)
        cursor.execute(maria_encounter_device_patient_update_trigger)

        # Stored Procedures
        cursor.execute(maria_insert_interval_stored_procedure)

        conn.commit()
        cursor.close()
        conn.close()

    def interval_exists(self, measure_id, device_id, start_time_nano):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_interval_exists_query, (measure_id, device_id, start_time_nano))
            result = cursor.fetchone()
        return result[0]

    def select_all_devices(self):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM device")
            rows = cursor.fetchall()
        return rows

    def select_all_measures(self):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM measure")
            rows = cursor.fetchall()
        return rows

    def select_all_patients(self):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM patient")
            rows = cursor.fetchall()
        return rows

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None):
        units = DEFAULT_UNITS if units is None else units

        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_insert_ignore_measure_query, (measure_tag, freq_nhz, units, measure_name))
            conn.commit()
            cursor.execute(maria_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
            measure_id = cursor.fetchone()[0]

        return measure_id

    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units
        with self.maria_db_connection() as (conn, cursor):
            if measure_id is not None:
                cursor.execute(maria_select_measure_from_id, (measure_id,))
            else:
                cursor.execute(maria_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
            row = cursor.fetchone()

        return row

    def insert_device(self, device_tag: str, device_name: str = None):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_insert_ignore_device_query, (device_tag, device_name))
            conn.commit()
            cursor.execute(maria_select_device_from_tag_query, (device_tag,))
            device_id = cursor.fetchone()[0]
        return device_id

    def select_device(self, device_id: int = None, device_tag: str = None):
        with self.maria_db_connection() as (conn, cursor):
            if device_id is not None:
                cursor.execute(maria_select_device_from_id_query, (device_id,))
            else:
                cursor.execute(maria_select_device_from_tag_query, (device_tag,))
            row = cursor.fetchone()
        return row

    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict]):
        with self.maria_db_connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute(maria_insert_file_index_query, (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            block_tuples = [(block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
                             block["start_time_n"], block["end_time_n"], block["num_values"])
                            for block in block_data]
            cursor.executemany(maria_insert_block_query, block_tuples)

            # insert into interval_index
            [cursor.callproc("insert_interval", (interval["measure_id"], interval["device_id"], interval["start_time_n"],
                                interval["end_time_n"])) for interval in interval_data]

    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int]):
        with self.maria_db_connection(begin=True) as (conn, cursor):
            # insert/update file data
            for file_path, (block_data, interval_data) in file_data.items():
                # insert file_path into file_index and get id
                cursor.execute(maria_insert_file_index_query, (file_path,))
                file_id = cursor.lastrowid

                # insert into block_index
                block_tuples = [
                    (block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
                     block["start_time_n"], block["end_time_n"], block["num_values"])
                    for block in block_data]
                cursor.executemany(maria_insert_block_query, block_tuples)

                # insert into interval_index
                if len(interval_data) > 0:
                    [cursor.callproc("insert_interval",
                                     (interval["measure_id"], interval["device_id"], interval["start_time_n"],
                                      interval["end_time_n"])) for interval in interval_data]

            # delete old block data
            cursor.executemany(maria_delete_block_query, [(block_id,) for block_id in block_ids_to_delete])

            # delete old file data (will delete later)
            # cursor.executemany(maria_delete_file_query, [(file_id,) for file_id in file_ids_to_delete])

    def select_file(self, file_id: int = None, file_path: str = None):
        with self.maria_db_connection() as (conn, cursor):
            if file_id is not None:
                cursor.execute(maria_select_file_by_id, (file_id,))
            else:
                cursor.execute(maria_select_file_by_values, (file_path,))
            row = cursor.fetchone()
        return row

    def select_files(self, file_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(file_id_list))
        maria_select_files_by_id_list = f"SELECT * FROM file_index WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            _LOGGER.debug(f"maria_select_files_by_id_list {maria_select_files_by_id_list}")
            _LOGGER.debug(f"file_id_list {file_id_list}")
            cursor.execute(maria_select_files_by_id_list, file_id_list)
            rows = cursor.fetchall()
        return rows

    def select_blocks_from_file(self, file_id: int):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_blocks_from_file, (file_id,))
            rows = cursor.fetchall()
        return rows

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

    def select_interval(self, interval_id: int = None, measure_id: int = None, device_id: int = None,
                        start_time_n: int = None, end_time_n: int = None):
        with self.maria_db_connection() as (conn, cursor):
            if interval_id is not None:
                cursor.execute(maria_select_interval_by_id, (interval_id,))
            else:
                cursor.execute(maria_select_interval_by_values, (measure_id, device_id, start_time_n, end_time_n))
            row = cursor.fetchone()
        return row

    def insert_setting(self, setting_name: str, setting_value: str):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(mariadb_setting_insert_query, (setting_name, setting_value))

    def select_setting(self, setting_name: str):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(mariadb_setting_select_query, (setting_name,))
            setting = cursor.fetchone()
        return setting

    def select_all_settings(self):
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_all_query)
            settings = cursor.fetchall()
        return settings

    def insert_source(self, s_name, description=None):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_source_query, (s_name, description))
            return cursor.lastrowid

    def insert_institution(self, i_name):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_institution_query, (i_name,))
            return cursor.lastrowid

    def insert_unit(self, institution_id, u_name, u_type):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_unit_query, (institution_id, u_name, u_type))
            return cursor.lastrowid

    def insert_bed(self, unit_id, b_name):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_bed_query, (unit_id, b_name))
            return cursor.lastrowid

    def insert_patient(self, patient_id=None, mrn=None, gender=None, dob=None, first_name=None, middle_name=None,
                       last_name=None, first_seen=None, last_updated=None, source_id=1):
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_insert_ignore_patient_query,
                           (patient_id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated,
                            source_id))
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

            with self.maria_db_connection(begin=False) as (conn, cursor):
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

        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def get_device_time_ranges_by_patient(self, patient_id, end_time_n, start_time_n):
        # patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ? AND end_time IS NOT NULL"
        patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ?"
        args = (patient_id,)
        if start_time_n is not None:
            patient_device_query += " AND end_time >= ? "
            args += (start_time_n,)
        if end_time_n is not None:
            patient_device_query += " AND start_time <= ? "
            args += (end_time_n,)
        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(patient_device_query, args)
            device_time_ranges = cursor.fetchall()
        return device_time_ranges

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

            with self.maria_db_connection(begin=False) as (conn, cursor):
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

        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(interval_query, args)
            return cursor.fetchall()

    def select_encounters(self, patient_id_list: List[int] = None, mrn_list: List[int] = None, start_time: int = None,
                          end_time: int = None):
        assert (patient_id_list is None) != (
                mrn_list is None), "Either patient_id_list or mrn_list must be provided, but not both"
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

    def select_all_measures_in_list(self, measure_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(measure_id_list))
        maria_select_measures_by_id_list = f"SELECT * FROM measure WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_measures_by_id_list, measure_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_patients_in_list(self, patient_id_list: List[int] = None, mrn_list: List[int] = None):
        if patient_id_list is not None:
            placeholders = ', '.join(['?'] * len(patient_id_list))
            maria_select_patients_by_id_list = f"SELECT * FROM patient WHERE id IN ({placeholders})"
        elif mrn_list is not None:
            patient_id_list = mrn_list
            placeholders = ', '.join(['?'] * len(patient_id_list))
            maria_select_patients_by_id_list = f"SELECT * FROM patient WHERE mrn IN ({placeholders})"
        else:
            maria_select_patients_by_id_list = "SELECT * FROM patient"
            patient_id_list = tuple()
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_patients_by_id_list, patient_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_devices_in_list(self, device_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(device_id_list))
        maria_select_devices_by_id_list = f"SELECT * FROM device WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_devices_by_id_list, device_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_beds_in_list(self, bed_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(bed_id_list))
        maria_select_beds_by_id_list = f"SELECT * FROM bed WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_beds_by_id_list, bed_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_units_in_list(self, unit_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(unit_id_list))
        maria_select_units_by_id_list = f"SELECT * FROM unit WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_units_by_id_list, unit_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_institutions_in_list(self, institution_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(institution_id_list))
        maria_select_institutions_by_id_list = f"SELECT * FROM institution WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_institutions_by_id_list, institution_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_device_encounters_by_encounter_list(self, encounter_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(encounter_id_list))
        maria_select_device_encounters_by_encounter_list = f"SELECT * FROM device_encounter WHERE encounter_id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_device_encounters_by_encounter_list, encounter_id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_sources_in_list(self, source_id_list: List[int]):
        placeholders = ', '.join(['?'] * len(source_id_list))
        maria_select_sources_by_id_list = f"SELECT * FROM source WHERE id IN ({placeholders})"
        with self.maria_db_connection() as (conn, cursor):
            cursor.execute(maria_select_sources_by_id_list, source_id_list)
            rows = cursor.fetchall()
        return rows

    def select_device_patients(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                               start_time: int = None, end_time: int = None):
        arg_tuple = ()
        maria_select_device_patient_query = \
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
        maria_select_device_patient_query += join_sql_and_bools(where_clauses)
        maria_select_device_patient_query += " ORDER BY id ASC"

        with self.maria_db_connection(begin=False) as (conn, cursor):
            cursor.execute(maria_select_device_patient_query, arg_tuple)
            return cursor.fetchall()

    def insert_device_patients(self, device_patient_data: List[Tuple[int, int, int, int]]):
        maria_insert_device_patient_query = \
            "INSERT INTO device_patient (device_id, patient_id, start_time, end_time) VALUES (?, ?, ?, ?)"

        with self.maria_db_connection() as (conn, cursor):
            cursor.executemany(maria_insert_device_patient_query, device_patient_data)
            conn.commit()

