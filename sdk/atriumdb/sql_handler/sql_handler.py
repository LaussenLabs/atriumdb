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
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

from atriumdb.sql_handler.sql_helper import join_sql_and_bools


class SQLHandler(ABC):
    @abstractmethod
    def create_schema(self):
        # Creates Tables if they don't exist.
        pass

    @abstractmethod
    def connection(self, begin: bool = False):
        pass

    @abstractmethod
    def select_all_devices(self):
        pass

    @abstractmethod
    def select_all_measures(self):
        pass

    @abstractmethod
    def select_all_patients(self):
        pass

    def select_patient_history(self, patient_id: int, field: Optional[str], start_time: int, end_time: int):
        # Dynamically construct the query based on whether `field` is None
        if field is None:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (int(patient_id), int(start_time), int(end_time))
        else:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND field = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (int(patient_id), field, int(start_time), int(end_time))

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, query_params)
            return cursor.fetchall()

    def select_closest_patient_history(self, patient_id: int, field: str, time: int):
        # Find the patient history that is closest to the timestamp
        query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? and field = ? and time <= ? ORDER BY time DESC LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (int(patient_id), field, int(time)))
            return cursor.fetchone()

    def select_unique_history_fields(self) -> List[str]:
        query = "SELECT DISTINCT field FROM patient_history"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query)
            # Fetch all results
            results = cursor.fetchall()
            # Extract field values from the results
            fields = [row[0] for row in results]
        return fields

    @abstractmethod
    def insert_measure(self, measure_tag: str, freq_nhz: int, units: Optional[str] = None, measure_name: Optional[str] = None,
                       measure_id: Optional[int] = None, code: Optional[str] = None, unit_label: Optional[str] = None, unit_code: Optional[str] = None,
                       source_id: Optional[int] = None):
        pass

    @abstractmethod
    def select_measure(self, measure_id: Optional[int] = None, measure_tag: Optional[str] = None, freq_nhz: Optional[int] = None, units: Optional[str] = None):
        # Select a measure either by its id, or by a tag, freq, units triplet.
        pass

    @abstractmethod
    def insert_device(self, device_tag: str, device_name: Optional[str] = None, device_id: Optional[int] = None, manufacturer: Optional[str] = None,
                      model: Optional[str] = None, device_type: Optional[str] = None, bed_id: Optional[int] = None, source_id: Optional[int] = None):
        pass

    @abstractmethod
    def select_device(self, device_id: Optional[int] = None, device_tag: Optional[str] = None):
        # Select a measure either by its id, or by its unique tag.
        pass

    @abstractmethod
    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode: str, gap_tolerance: int = 0):
        # Insert a file path to file index.
        # Insert block_index rows with foreign key file_id.
        # Insert interval_index rows.
        pass

    def insert_and_delete_tsc_file_data(self, file_path: str, block_data: List[Dict], block_ids_to_delete: List[int]):
        with self.connection(begin=True) as (conn, cursor):
            # Insert file_path into file_index and get id
            cursor.execute("INSERT INTO file_index (path) VALUES (?);", (file_path,))
            file_id = cursor.lastrowid

            # Insert into block_index
            block_tuples = [(block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
                             block["start_time_n"], block["end_time_n"], block["num_values"])
                            for block in block_data]

            insert_block_query = """INSERT INTO block_index (measure_id, device_id, file_id, start_byte, num_bytes, 
                start_time_n, end_time_n, num_values) VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.executemany(insert_block_query, block_tuples)

            # Delete blocks with matching block ids in the block_ids list
            if block_ids_to_delete:
                delete_query = "DELETE FROM block_index WHERE id = ?;"
                cursor.executemany(delete_query, [(block_id,) for block_id in block_ids_to_delete])

    def replace_intervals(self, measure_id: int, device_id: int, interval_list: List[List[int]]):
        if len(interval_list) == 0:
            raise ValueError("This function deletes and replaces all intervals. `interval_list` cannot be empty")

        with self.connection(begin=True) as (conn, cursor):
            # Delete existing intervals for the given measure_id and device_id
            delete_query = """
                DELETE FROM interval_index
                WHERE measure_id = ? AND device_id = ?;
            """
            cursor.execute(delete_query, (measure_id, device_id))

            # Insert new intervals into the interval_index table
            insert_query = """
                INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
                VALUES (?, ?, ?, ?);
            """
            interval_tuples = [(int(measure_id), int(device_id), int(start_time), int(end_time))
                               for (start_time, end_time) in interval_list]
            cursor.executemany(insert_query, interval_tuples)

    @abstractmethod
    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int], gap_tolerance: int = 0):
        pass

    @abstractmethod
    def insert_merged_block_data(self, file_path: str, block_data: List[Dict], old_block_id: int, interval_data: List[Dict],
                                 interval_index_mode: str, gap_tolerance: int = 0):
        pass

    @abstractmethod
    def select_file(self, file_id: Optional[int] = None, file_path: Optional[str] = None):
        # Select a file path either by its id, or by its path.
        pass

    @abstractmethod
    def select_files(self, file_id_list: List[int]):
        # Selects all files from list of ids.
        pass

    @abstractmethod
    def select_blocks_from_file(self, file_id: int):
        # Selects all blocks from file_id
        pass

    @abstractmethod
    def select_block(self, block_id: Optional[int] = None, measure_id: Optional[int] = None, device_id: Optional[int] = None, file_id: Optional[int] = None,
                     start_byte: Optional[int] = None, num_bytes: Optional[int] = None, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None,
                     num_values: Optional[int] = None):
        # Select a block either by its id or by all other params.
        pass

    def select_closest_block(self, measure_id: int, device_id: int, start_time: int, end_time: int):
        base_query = """SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values 
        FROM block_index WHERE measure_id = ? and device_id = ? """

        with self.connection(begin=False) as (conn, cursor):
            # First check if this block belongs on the end (most likely scenario)
            end_check_query = base_query + "ORDER BY end_time_n DESC LIMIT 1"
            cursor.execute(end_check_query, (int(measure_id), int(device_id)))
            result = cursor.fetchone()

            # Check if the start time is >= the max end time meaning the block goes on the end
            if result is not None and int(start_time) >= result[7]:
                # Return true meaning it goes on the end. We will check if the last block is full in write_data so we
                # need to know if this is an end block
                return result, True

            # If it overlaps with the last block
            if result is not None and int(start_time) <= result[7] and int(end_time) >= result[6]:
                return result, False

            # Check if there is a block that this data fits inside
            inside_query = base_query + "and start_time_n <= ? and end_time_n >= ? LIMIT 1"

            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            result = cursor.fetchone()

            # If there is a block this data belongs inside return it
            if result is not None:
                return result, False

            # Get the closest block whose start time is <= my start time
            inside_query = base_query + "and start_time_n <= ? and end_time_n <= ? ORDER BY start_time_n DESC, end_time_n DESC LIMIT 1"
            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            block_older = cursor.fetchone()

            # Get the closest block whose end time is >= my end time
            inside_query = base_query + "and start_time_n >= ? and end_time_n >= ? ORDER BY end_time_n ASC, start_time_n ASC LIMIT 1"
            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            block_newer = cursor.fetchone()

            # Subtract the end time of the old block from the start time of the new block to see how far apart they are
            # If they overlap the number will become negative and therefore they are closer
            older_diff = int(start_time) - block_older[7] if block_older is not None else None

            # Subtract the start time of the old block from the end time of the new block to see how far apart they are
            # If they overlap the number will become negative and therefore they are closer
            newer_diff = block_newer[6] - int(end_time) if block_newer is not None else None

            if older_diff is None and newer_diff is not None:
                return block_newer, False
            elif newer_diff is None and older_diff is not None:
                return block_older, False
            elif older_diff is None and newer_diff is None:
                # Need this since if it's the first block there will be nothing to merge with
                return None, False
            elif older_diff <= newer_diff:
                return block_older, False
            elif older_diff > newer_diff:
                return block_newer, False

    def delete_block(self, block_id: int):
        with self.connection(begin=True) as (conn, cursor):
            # Delete block data
            cursor.execute("DELETE FROM block_index WHERE id = ?", (int(block_id),))

    def select_blocks_for_device(self, device_id: int, measure_ids: int|List[int] = None):
        """
        Fetch block index data for the device (and measures if specified).
        """
        # Normalize measure_ids to a list if it's an integer
        if measure_ids is not None:
            if isinstance(measure_ids, int):
                measure_ids = [measure_ids]
            elif not isinstance(measure_ids, list):
                raise TypeError("measure_ids must be an int, a list of ints, or None.")

            # Build placeholders for SQL IN clause
            placeholders = ','.join(['?'] * len(measure_ids))
            block_query = f"""
            SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
            FROM block_index
            WHERE device_id = ? AND measure_id IN ({placeholders})
            ORDER BY measure_id, device_id, start_time_n ASC;
            """
            args = [int(device_id)] + [int(mid) for mid in measure_ids]
        else:
            # Fetch blocks for the device across all measures
            block_query = """
            SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
            FROM block_index
            WHERE device_id = ?
            ORDER BY measure_id, device_id, start_time_n ASC;
            """
            args = [int(device_id)]

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def select_blocks_for_devices(self, device_ids: List[int], measure_ids: List[int]):
        # Build placeholders for SQL IN clauses
        device_placeholders = ','.join(['?'] * len(device_ids))
        measure_placeholders = ','.join(['?'] * len(measure_ids))

        block_query = f"""
        SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
        FROM block_index
        WHERE device_id IN ({device_placeholders}) AND measure_id IN ({measure_placeholders})
        ORDER BY measure_id, device_id, start_time_n ASC;
        """

        # Prepare arguments for the SQL query
        args = [int(did) for did in device_ids] + [int(mid) for mid in measure_ids]

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    @abstractmethod
    def select_interval(self, interval_id: Optional[int] = None, measure_id: Optional[int] = None, device_id: Optional[int] = None,
                        start_time_n: Optional[int] = None, end_time_n: Optional[int] = None):
        # Select an interval either by its id, or by all other params.
        pass

    @abstractmethod
    def insert_setting(self, setting_name: str, setting_value: str):
        # Inserts a setting into the database.
        pass

    @abstractmethod
    def select_setting(self, setting_name: str):
        # Selects a setting from the database.
        pass

    @abstractmethod
    def insert_patient(self, patient_id: Optional[int] = None, mrn: Optional[str] = None, gender: Optional[str] = None, dob: Optional[str] = None,
                       first_name: Optional[str] = None, middle_name: Optional[str] = None, last_name: Optional[str] = None, first_seen: Optional[int] = None,
                       last_updated: Optional[int] = None, source_id: int = 1, weight: Optional[float] = None, height: Optional[float] = None):
        # Insert patient if it doesn't exist, return id.
        pass

    def insert_patient_history(self, patient_id: int, field: str, value: float, units: str, time: int):
        with self.connection(begin=True) as (conn, cursor):
            # Find the most recent value for the field you're entering from the patient history table
            cursor.execute("SELECT MAX(time) FROM patient_history WHERE patient_id = ? and field = ?", (int(patient_id), field))
            newest_measurement_time = cursor.fetchone()[0]

            # If the new measurement is newer than the newest in the patient history table, update the field in the
            # patient table. If no history is found, also update it.
            if newest_measurement_time is None or int(time) > newest_measurement_time:
                cursor.execute(f"UPDATE patient SET {field} = {value} WHERE id = {int(patient_id)}")

            # Now insert the row to the patient history table
            query = "INSERT INTO patient_history (patient_id, field, value, units, time) VALUES (?, ?, ?, ?, ?)"
            cursor.execute(query, (int(patient_id), field, float(value), units, int(time)))
            conn.commit()
            return cursor.lastrowid

    @abstractmethod
    def insert_encounter(self, patient_id: int, bed_id: int, start_time: int, end_time: Optional[int] = None, source_id: int = 1,
                         visit_number: Optional[int] = None, last_updated: Optional[int] = None):
        # Insert encounter if it doesn't exist, return id.
        pass

    @abstractmethod
    def insert_device_encounter(self, device_id: int, encounter_id: int, start_time: int, end_time: Optional[int] = None,
                                source_id: int = 1):
        # Insert device_encounter if it doesn't exist, return id.
        pass

    def get_device_time_ranges_by_patient(self, patient_id: int, end_time_n: Optional[int], start_time_n: Optional[int]):
        patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ?"
        args = (int(patient_id),)

        if start_time_n is not None:
            patient_device_query += " AND (end_time >= ? OR end_time is NULL) "
            args += (int(start_time_n),)
        if end_time_n is not None:
            patient_device_query += " AND start_time <= ? "
            args += (int(end_time_n),)

        patient_device_query += " ORDER BY start_time, end_time"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(patient_device_query, args)
            return cursor.fetchall()

    @abstractmethod
    def select_all_settings(self):
        # Select all settings from settings table.
        pass

    @abstractmethod
    def select_blocks(self, measure_id: int, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None, device_id: Optional[int] = None, patient_id: Optional[int] = None):
        # Get all matching blocks.
        pass

    def select_intervals(self, measure_id: int, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None, device_id: Optional[int] = None, patient_id: Optional[int] = None):
        if device_id is None and patient_id is None:
            raise ValueError("Either device_id or patient_id must be provided")

        interval_query = "SELECT id, measure_id, device_id, start_time_n, end_time_n FROM interval_index WHERE measure_id = ? AND device_id = ?"

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)
            # Add start and end time to the query
            interval_query += " AND end_time_n >= ? AND start_time_n <= ? ORDER BY start_time_n ASC, end_time_n ASC"

            interval_results = []
            with self.connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    encounter_end_time = time.time_ns() if encounter_end_time is None else encounter_end_time
                    args = (int(measure_id), int(encounter_device_id), int(encounter_start_time), int(encounter_end_time))

                    cursor.execute(interval_query, args)

                    #  Truncate Intervals to the Start, End of Encounter
                    encounter_intervals = cursor.fetchall()
                    encounter_intervals = [[interval_id,
                                            measure_id,
                                            device_id,
                                            max(start_time_n, encounter_start_time),
                                            min(end_time_n, encounter_end_time)]
                                           for interval_id, measure_id, device_id, start_time_n, end_time_n
                                           in encounter_intervals]

                    interval_results.extend(encounter_intervals)

            return interval_results

        # Query by device.
        args = (int(measure_id), int(device_id))

        if end_time_n is not None:
            interval_query += " AND start_time_n <= ?"
            args += (int(end_time_n),)

        if start_time_n is not None:
            interval_query += " AND end_time_n >= ?"
            args += (int(start_time_n),)

        # Add the ordering
        interval_query += " ORDER BY start_time_n ASC, end_time_n ASC"

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(interval_query, args)
            return cursor.fetchall()

    @abstractmethod
    def select_encounters(self, patient_id_list: Optional[List[int]] = None, mrn_list: Optional[List[int]] = None, start_time: Optional[int] = None,
                          end_time: Optional[int] = None):
        # Get all matching encounters.
        pass

    @abstractmethod
    def select_all_measures_in_list(self, measure_id_list: List[int]):
        # Get all matching measures.
        pass

    @abstractmethod
    def select_all_patients_in_list(self, patient_id_list: Optional[List[int]] = None, mrn_list: Optional[List[int]] = None):
        # Get all matching patients.
        pass

    @abstractmethod
    def select_all_devices_in_list(self, device_id_list: List[int]):
        # Get all matching devices.
        pass

    @abstractmethod
    def select_all_beds_in_list(self, bed_id_list: List[int]):
        # Get all matching beds.
        pass

    @abstractmethod
    def select_all_units_in_list(self, unit_id_list: List[int]):
        # Get all matching units.
        pass

    @abstractmethod
    def select_all_institutions_in_list(self, institution_id_list: List[int]):
        # Get all matching institutions.
        pass

    @abstractmethod
    def select_all_device_encounters_by_encounter_list(self, encounter_id_list: List[int]):
        # Get all matching device_encounters by encounter id list.
        pass

    @abstractmethod
    def select_all_sources_in_list(self, source_id_list: List[int]):
        # Get all matching sources.
        pass

    def select_device_patients(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                               start_time: int = None, end_time: int = None):
        arg_tuple = ()
        sqlite_select_device_patient_query = \
            "SELECT device_id, patient_id, start_time, end_time FROM device_patient"
        where_clauses = []

        # Handle device_id_list
        if device_id_list is not None and len(device_id_list) > 0:
            where_clauses.append("device_id IN ({})".format(
                ','.join(['?'] * len(device_id_list))))
            arg_tuple += tuple(int(device_id) for device_id in device_id_list)

        # Handle patient_id_list
        if patient_id_list is not None and len(patient_id_list) > 0:
            where_clauses.append("patient_id IN ({})".format(
                ','.join(['?'] * len(patient_id_list))))
            arg_tuple += tuple(int(patient_id) for patient_id in patient_id_list)

        # Handle start_time
        if start_time is not None:
            where_clauses.append("(end_time > ? OR end_time IS NULL)")
            arg_tuple += (int(start_time),)

        # Handle end_time
        if end_time is not None:
            where_clauses.append("start_time < ?")
            arg_tuple += (int(end_time),)

        # Combine where clauses
        if where_clauses:
            sqlite_select_device_patient_query += " WHERE " + " AND ".join(where_clauses)

        sqlite_select_device_patient_query += " ORDER BY id ASC"

        with self.connection() as (conn, cursor):
            cursor.execute(sqlite_select_device_patient_query, arg_tuple)
            return cursor.fetchall()

    def select_device_patient_encounters(self, timestamp: int, device_id_list: List[int] = None,
                                         patient_id_list: List[int] = None):
        arg_tuple = (timestamp, timestamp)
        sql_select_query = (
            "SELECT device_id, patient_id, start_time, end_time "
            "FROM device_patient WHERE start_time <= ? AND (end_time > ? OR end_time IS NULL)"
        )

        where_clauses = []

        # Handle device_id_list
        if device_id_list is not None and len(device_id_list) > 0:
            where_clauses.append("device_id IN ({})".format(
                ','.join(['?'] * len(device_id_list))))
            arg_tuple += tuple(int(device_id) for device_id in device_id_list)

        # Handle patient_id_list
        if patient_id_list is not None and len(patient_id_list) > 0:
            where_clauses.append("patient_id IN ({})".format(
                ','.join(['?'] * len(patient_id_list))))
            arg_tuple += tuple(int(patient_id) for patient_id in patient_id_list)

        # Combine where clauses
        if where_clauses:
            sql_select_query += " AND " + " AND ".join(where_clauses)

        sql_select_query += " ORDER BY id DESC"

        with self.connection() as (conn, cursor):
            cursor.execute(sql_select_query, arg_tuple)
            results = cursor.fetchall()
            return results

    def insert_encounter_row(self, patient_id: int, bed_id: int, start_time: int, end_time: int = None,
                         source_id: int = 1, visit_number: str = None, last_updated: int = None):
        query = """
            INSERT INTO encounter (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.connection() as (conn, cursor):
            cursor.execute(query, (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated))
            conn.commit()

    def select_encounters_from_range_or_timestamp(
            self, timestamp: int = None, start_time: int = None, end_time: int = None,
            bed_id: int = None, patient_id: int = None):
        query = "SELECT id, patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated FROM encounter"
        where_clauses = []
        args = []

        if timestamp is not None:
            where_clauses.append("start_time <= ? AND (end_time > ? OR end_time IS NULL)")
            args.extend([timestamp, timestamp])

        if start_time is not None:
            where_clauses.append("(end_time > ? OR end_time IS NULL)")
            args.append(start_time)

        if end_time is not None:
            where_clauses.append("start_time < ?")
            args.append(end_time)

        if bed_id is not None:
            where_clauses.append("bed_id = ?")
            args.append(bed_id)

        if patient_id is not None:
            where_clauses.append("patient_id = ?")
            args.append(patient_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY start_time ASC"

        with self.connection() as (conn, cursor):
            cursor.execute(query, args)
            return cursor.fetchall()

    @abstractmethod
    def insert_device_patients(self, device_patient_data: List[Tuple[int, int, int, int]]):
        # Insert device_patient rows.
        pass

    def insert_label_set(self, name: str, label_set_id: Optional[int] = None, parent_id: Optional[int] = None):
        if label_set_id is not None:
            existing_label_set = self.select_label_set(label_set_id)
            if existing_label_set:
                if existing_label_set[1] == name and existing_label_set[2] == parent_id:
                    # The provided ID exists and matches the name
                    return label_set_id
                else:
                    # The provided ID exists but with a different name
                    raise ValueError(f"The id {label_set_id} already exists under label name {existing_label_set[1]} "
                                     f"and parent {existing_label_set[2]}")

        existing_label_set_id = self.select_label_set_id(name)
        if existing_label_set_id is not None:
            # The name already exists with a different ID
            return existing_label_set_id

        # Insert the new label set
        query = "INSERT INTO label_set (id, name, parent_id) VALUES (?, ?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.execute(query, (label_set_id, name, parent_id,))
            conn.commit()
            return cursor.lastrowid if label_set_id is None else label_set_id

    @abstractmethod
    def select_label_sets(self):
        # Retrieve all label types
                pass

    def select_label_set(self, label_set_id: int):
        query = "SELECT id, name, parent_id FROM label_set WHERE id = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id),))
            row = cursor.fetchone()
        return row

    def select_label_set_id(self, name: str) -> Optional[int]:
        # Retrieve the ID of a label type by its name.
        query = "SELECT id FROM label_set WHERE name = ? LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            # Return the ID if it exists or None otherwise.
            return result[0] if result else None

    def select_label_name_parent(self, label_set_id: int) -> Optional[Tuple[int, str]]:
        query = """
        SELECT parent.id, parent.name FROM label_set
        INNER JOIN label_set AS parent ON label_set.parent_id = parent.id
        WHERE {}
        LIMIT 1;
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query.format("label_set.id = ?"), (int(label_set_id),))
            return cursor.fetchone()

    def select_all_ancestors(self, label_set_id: Optional[int] = None, name: Optional[str] = None) -> Optional[List[Tuple[int, str]]]:
        if label_set_id is None and name is None:
            raise ValueError("Either label_set_id or name must be provided")

        query = """
        WITH RECURSIVE ancestors(id, name, parent_id) AS (
            SELECT id, name, parent_id FROM label_set WHERE {}
            UNION ALL
            SELECT ls.id, ls.name, ls.parent_id FROM label_set ls
            INNER JOIN ancestors ON ls.id = ancestors.parent_id
        )
        SELECT id, name FROM ancestors WHERE id != ?;
        """

        with self.connection() as (conn, cursor):
            if label_set_id is not None:
                cursor.execute(query.format("id = ?"), (int(label_set_id), int(label_set_id)))
            else:
                query_for_name = """
                SELECT id FROM label_set WHERE name = ? LIMIT 1;
                """
                cursor.execute(query_for_name, (name,))
                row = cursor.fetchone()
                if row:
                    cursor.execute(query.format("id = ?"), (row[0], row[0]))
                else:
                    return None
            return cursor.fetchall()

    def select_label_name_children(self, label_set_id: int) -> List[Tuple[int, str]]:
        query = """
        SELECT id, name FROM label_set
        WHERE parent_id = ?
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id),))
            return cursor.fetchall()

    def select_all_label_name_descendents(self, label_set_id: int):

        query = """
        WITH RECURSIVE descendants(id, name, parent_id) AS (
            SELECT id, name, parent_id FROM label_set WHERE parent_id = ?
            UNION ALL
            SELECT ls.id, ls.name, ls.parent_id FROM label_set ls
            INNER JOIN descendants ON ls.parent_id = descendants.id
        )
        SELECT id, name, parent_id FROM descendants WHERE id != ?;
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id), int(label_set_id)))
            return cursor.fetchall()

    def insert_label(self, label_set_id, device_id, start_time_n, end_time_n, label_source_id=None, measure_id=None):
        # Insert a new label record into the database.
        query = "INSERT INTO label (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n) VALUES (?, ?, ?, ?, ?, ?)"
        with self.connection() as (conn, cursor):
            label_set_id = None if label_set_id is None else int(label_set_id)
            device_id = None if device_id is None else int(device_id)
            start_time_n = None if start_time_n is None else int(start_time_n)
            end_time_n = None if end_time_n is None else int(end_time_n)
            label_source_id = None if label_source_id is None else int(label_source_id)
            measure_id = None if measure_id is None else int(measure_id)
            cursor.execute(query, (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n))
            conn.commit()
            # Return the ID of the newly inserted label.
            return cursor.lastrowid

    def insert_labels(self, labels):
        # Insert multiple label records into the database.
        formatted_labels = []
        for label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n in labels:
            label_set_id = None if label_set_id is None else int(label_set_id)
            device_id = None if device_id is None else int(device_id)
            start_time_n = None if start_time_n is None else int(start_time_n)
            end_time_n = None if end_time_n is None else int(end_time_n)
            label_source_id = None if label_source_id is None else int(label_source_id)
            measure_id = None if measure_id is None else int(measure_id)
            formatted_labels.append([label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n])
        query = "INSERT INTO label (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n) VALUES (?, ?, ?, ?, ?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.executemany(query, formatted_labels)
            conn.commit()
            # Return the ID of the last inserted label.
            return cursor.lastrowid

    def delete_labels(self, label_ids):
        # Delete multiple label records from the database based on their IDs.
        query = "DELETE FROM label WHERE id = ?"
        with self.connection() as (conn, cursor):
            # Prepare a list of tuples for the executemany method.
            id_tuples = [(int(label_id),) for label_id in label_ids]
            cursor.executemany(query, id_tuples)
            conn.commit()

    def select_labels(self, label_set_id_list=None, device_id_list=None, patient_id_list=None, start_time_n=None,
                      end_time_n=None, label_source_id_list=None, measure_id_list=None, limit=None, offset=None):
        if device_id_list is not None and patient_id_list is not None:
            raise ValueError("Can only request labels by device or patient, not both")

        # If provided patient IDs, fetch device time ranges and recursively call select_labels.
        if patient_id_list is not None:
            results = []
            for patient_id in patient_id_list:
                # Get device time ranges associated with a patient.
                device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

                for device_id, device_start_time, device_end_time in device_time_ranges:
                    # Adjust the time range based on the provided boundaries.
                    final_start_time = max(start_time_n, device_start_time) if start_time_n else device_start_time
                    final_end_time = min(end_time_n, device_end_time) if end_time_n else device_end_time

                    # Recursively fetch labels for each device and accumulate the results.
                    results.extend(self.select_labels(label_set_id_list=label_set_id_list, device_id_list=[device_id],
                                                      start_time_n=final_start_time, end_time_n=final_end_time,
                                                      label_source_id_list=label_source_id_list, measure_id_list=measure_id_list))

            # Sort the results by start_time_n primarily and then by end_time_n secondarily
            results.sort(key=lambda x: (x[5], x[6], x[0]))
            return results

        # Construct the query for selecting labels based on the provided criteria.
        query = "SELECT id, label_set_id, device_id, measure_id, label_source_id , start_time_n, end_time_n FROM label WHERE 1=1"
        params = []

        # Add conditions for label type IDs, if provided.
        if label_set_id_list:
            placeholders = ', '.join(['?'] * len(label_set_id_list))
            query += f" AND label_set_id IN ({placeholders})"
            params.extend(label_set_id_list)

        # Add conditions for device IDs, if provided.
        if device_id_list:
            placeholders = ', '.join(['?'] * len(device_id_list))
            query += f" AND device_id IN ({placeholders})"
            params.extend(device_id_list)

        # Add conditions for measure IDs, if provided.
        if measure_id_list:
            placeholders = ', '.join(['?'] * len(measure_id_list))
            query += f" AND measure_id IN ({placeholders})"
            params.extend(measure_id_list)

        # Add conditions for label source IDs, if provided.
        if label_source_id_list:
            placeholders = ', '.join(['?'] * len(label_source_id_list))
            query += f" AND label_source_id IN ({placeholders})"
            params.extend(label_source_id_list)

        # Add conditions for start and end times, if provided.
        if start_time_n:
            query += " AND end_time_n >= ?"
            params.append(int(start_time_n))
        if end_time_n:
            query += " AND start_time_n <= ?"
            params.append(int(end_time_n))

        # Sort by start_time_n
        # Used in iterator logic, alter with caution.
        query += " ORDER BY start_time_n ASC, end_time_n ASC, label.id ASC"

        # if limit and offset are specified add them to query
        if limit is not None and offset is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        # if only limit is supplied then only add it to the query
        elif limit is not None and offset is None:
            query += f" LIMIT {limit}"

        # Execute the query and return the results.
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchall()


    def select_labels_with_info(self, label_set_id_list=None, device_id_list=None, patient_id_list=None,
                                start_time_n=None, end_time_n=None, label_source_id_list=None, measure_id_list=None,
                                limit=None, offset=None):
        if device_id_list is not None and patient_id_list is not None:
            raise ValueError("Can only request labels by device or patient, not both")

        results = []

        if patient_id_list is not None:
            # Fetch device time ranges associated with a patient and adjust the time range based on provided boundaries.
            for patient_id in patient_id_list:
                device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

                for device_id, device_start_time, device_end_time in device_time_ranges:
                    final_start_time = max(start_time_n, device_start_time) if start_time_n else device_start_time
                    final_end_time = min(end_time_n, device_end_time) if end_time_n else device_end_time

                    # Recursively fetch labels and accumulate the results.
                    results.extend(
                        self.select_labels_with_info(label_set_id_list=label_set_id_list, device_id_list=[device_id],
                                                     start_time_n=final_start_time, end_time_n=final_end_time,
                                                     label_source_id_list=label_source_id_list,
                                                     measure_id_list=measure_id_list))

            # Sort the results by start_time_n primarily and then by end_time_n secondarily
            results.sort(key=lambda x: (x[7], x[8], x[0]))
            return results

        # Construct the query with additional joins for label_source, label_set, and device_patient.
        query = """
            SELECT
                label.id, label_set.name AS label_name, label_set_id, 
                label.device_id, label.measure_id, label_source.name AS label_source_name, 
                label_source_id, start_time_n, end_time_n, 
                device_patient.patient_id
            FROM label
            JOIN label_set ON label.label_set_id = label_set.id
            LEFT JOIN label_source ON label.label_source_id = label_source.id
            LEFT JOIN device_patient ON label.device_id = device_patient.device_id
                AND label.start_time_n >= device_patient.start_time
                AND label.end_time_n <= device_patient.end_time
        """
        params = []

        # Add conditions based on the provided criteria.
        conditions = ["1=1"]  # Placeholder for dynamic query construction

        if label_set_id_list:
            placeholders = ', '.join(['?'] * len(label_set_id_list))
            conditions.append(f"label.label_set_id IN ({placeholders})")
            params.extend(label_set_id_list)

        if device_id_list:
            placeholders = ', '.join(['?'] * len(device_id_list))
            conditions.append(f"label.device_id IN ({placeholders})")
            params.extend(device_id_list)

        if measure_id_list:
            placeholders = ', '.join(['?'] * len(measure_id_list))
            conditions.append(f"label.measure_id IN ({placeholders})")
            params.extend(measure_id_list)

        if label_source_id_list:
            placeholders = ', '.join(['?'] * len(label_source_id_list))
            conditions.append(f"label.label_source_id IN ({placeholders})")
            params.extend(label_source_id_list)

        if start_time_n:
            conditions.append("label.end_time_n >= ?")
            params.append(int(start_time_n))

        if end_time_n:
            conditions.append("label.start_time_n <= ?")
            params.append(int(end_time_n))

        # Combine conditions into the query.
        query += " WHERE " + " AND ".join(conditions)

        # Add sorting.
        query += " ORDER BY start_time_n ASC, end_time_n ASC, label.id ASC"

        # Handle limit and offset.
        if limit is not None:
            query += f" LIMIT {limit}"
            if offset is not None:
                query += f" OFFSET {offset}"

        # Execute the query and return the results.
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, params)
            fetched_results = cursor.fetchall()

        return fetched_results

    def insert_label_source(self, name, description=None):
        # First, check if the label_source with the given name already exists
        select_query = "SELECT id FROM label_source WHERE name = ?"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(select_query, (name,))
            result = cursor.fetchone()
            if result:
                # A label_source with the given name already exists, return its id
                return result[0]

        # If not found, insert the new label_source
        insert_query = "INSERT INTO label_source (name, description) VALUES (?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.execute(insert_query, (name, description))
            conn.commit()
            return cursor.lastrowid

    def select_label_source_id_by_name(self, name):
        query = "SELECT id FROM label_source WHERE name = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def select_label_source_info_by_id(self, label_source_id):
        query = "SELECT id, name, description FROM label_source WHERE id = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_source_id),))
            result = cursor.fetchone()
            return {'id': result[0], 'name': result[1], 'description': result[2]} if result else None
        pass

    @abstractmethod
    def select_all_label_sources(self, limit=None, offset=None):
        pass

    def get_measure_id_with_most_rows(self, tag: str) -> Optional[int]:
        # Query to get all matching measure.ids
        measure_ids_query = """
        SELECT id FROM measure WHERE tag = ?
        """
        measure_ids = []

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(measure_ids_query, (tag,))
            measure_ids = [row[0] for row in cursor.fetchall()]

        # Query to find the measure.id with the most rows in block_index
        most_rows_query = """
        SELECT measure_id, COUNT(*) as row_count
        FROM block_index
        WHERE measure_id IN ({})
        GROUP BY measure_id
        ORDER BY row_count DESC
        LIMIT 1
        """.format(','.join(['?'] * len(measure_ids)))

        if not measure_ids:
            return None

        if len(measure_ids) == 1:
            return measure_ids[0]
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(most_rows_query, measure_ids)
            result = cursor.fetchone()
            return result[0] if result else None

    def get_tag_to_measure_ids_dict(self, approx: bool = True) -> Dict[str, List[int]]:
        # Retrieve all measures and construct id-to-tag mapping
        measure_query = """
        SELECT id, tag FROM measure
        """
        id_to_tag = {}

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(measure_query)
            for row in cursor.fetchall():
                measure_id, tag = row
                if tag not in id_to_tag:
                    id_to_tag[tag] = []
                id_to_tag[tag].append(int(measure_id))

        # Get count of rows for each measure ID from block_index
        if approx:
            block_index_query = """
            SELECT measure_id, COUNT(*) as approx_count
            FROM block_index
            WHERE id <= 100000
            GROUP BY measure_id;
            """
        else:
            block_index_query = """
            SELECT measure_id, COUNT(*) as row_count
            FROM block_index
            GROUP BY measure_id
            """

        measure_id_to_count: Dict[int, int] = {}
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_index_query)
            for row in cursor.fetchall():
                measure_id, count = row
                measure_id_to_count[int(measure_id)] = count

        # Construct the final dictionary
        tag_to_sorted_measure_ids = {}
        for tag, measure_ids in id_to_tag.items():
            # Sort the measure IDs by count, descending order
            sorted_measure_ids = sorted(
                measure_ids,
                key=lambda x: measure_id_to_count.get(x, 0),
                reverse=True
            )
            tag_to_sorted_measure_ids[tag] = sorted_measure_ids

        return tag_to_sorted_measure_ids

    def insert_source(self, name: str, description: Optional[str] = None) -> int:
        query = "INSERT INTO source (name, description) VALUES (?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name, description))
            conn.commit()
            return cursor.lastrowid

    def select_source(self, source_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, str, Optional[str]]]:
        query = "SELECT id, name, description FROM source WHERE "
        params = []
        if source_id:
            query += "id = ?"
            params.append(int(source_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either source_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_institution(self, name: str) -> int:
        query = "INSERT INTO institution (name) VALUES (?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name,))
            conn.commit()
            return cursor.lastrowid

    def select_institution(self, institution_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, str]]:
        query = "SELECT id, name FROM institution WHERE "
        params = []
        if institution_id:
            query += "id = ?"
            params.append(int(institution_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either institution_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_unit(self, institution_id: int, name: str, unit_type: str) -> int:
        query = "INSERT INTO unit (institution_id, name, type) VALUES (?, ?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(institution_id), name, unit_type))
            conn.commit()
            return cursor.lastrowid

    def select_unit(self, unit_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, int, str, str]]:
        query = "SELECT id, institution_id, name, type FROM unit WHERE "
        params = []
        if unit_id:
            query += "id = ?"
            params.append(int(unit_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either unit_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_bed(self, unit_id: int, name: str, bed_id: Optional[int] = None) -> int:
        if bed_id is not None:
            query = "INSERT INTO bed (id, unit_id, name) VALUES (?, ?, ?)"
            params = (int(bed_id), int(unit_id), name)
        else:
            query = "INSERT INTO bed (unit_id, name) VALUES (?, ?)"
            params = (int(unit_id), name)

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            conn.commit()
            return bed_id if bed_id is not None else cursor.lastrowid

    def select_bed(self, bed_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, int, str]]:
        query = "SELECT id, unit_id, name FROM bed WHERE "
        params = []
        if bed_id:
            query += "id = ?"
            params.append(int(bed_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either bed_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def find_unreferenced_tsc_files(self):
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT t1.* FROM file_index t1 LEFT JOIN (SELECT DISTINCT file_id FROM block_index) t2 "
                           "ON t1.id = t2.file_id WHERE t2.file_id IS NULL")
            return cursor.fetchall()


    def delete_files_by_ids(self, file_ids_to_delete: List[int | tuple]):
        if len(file_ids_to_delete) == 0:
            return
        if isinstance(file_ids_to_delete[0], int):
            file_ids_to_delete = [(file_id,) for file_id in file_ids_to_delete]

        with self.connection(begin=False) as (conn, cursor):
            # if you put too many rows in the delete statement mariadb will fail. So we split it up
            for i in range(math.ceil(len(file_ids_to_delete) / 100_000)):
                # delete old tsc files
                cursor.executemany("DELETE FROM file_index WHERE id = ?;", file_ids_to_delete[i * 100_000:(i + 1) * 100_000])
