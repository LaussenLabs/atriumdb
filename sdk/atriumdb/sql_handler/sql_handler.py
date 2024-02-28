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

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class SQLHandler(ABC):
    @abstractmethod
    def create_schema(self):
        # Creates Tables if they dont exist.
        pass

    @abstractmethod
    def connection(self, begin=False):
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

    def select_patient_history(self, patient_id, field, start_time, end_time):
        # Dynamically construct the query based on whether `field` is None
        if field is None:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (patient_id, start_time, end_time)
        else:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND field = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (patient_id, field, start_time, end_time)

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, query_params)
            return cursor.fetchall()

    def select_closest_patient_history(self, patient_id, field, time):
        # find the patient history that is closest to the timestamp
        query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? and field = ? and time <= ? ORDER BY time DESC LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (patient_id, field, time))
            return cursor.fetchone()

    def select_unique_history_fields(self):
        query = "SELECT DISTINCT field FROM patient_history"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query)
            # Fetch all results
            results = cursor.fetchall()
            # Extract field values from the results
            fields = [row[0] for row in results]
        return fields

    @abstractmethod
    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None,
                       measure_id=None, code: str = None, unit_label: str = None, unit_code: str = None,
                       source_id: int = None):
        pass

    @abstractmethod
    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        # Select a measure either by its id, or by a tag, freq, units triplet.
        pass

    @abstractmethod
    def insert_device(self, device_tag: str, device_name: str = None, device_id=None, manufacturer: str = None,
                      model: str = None, device_type: str = None, bed_id: int = None, source_id: int = None):
        pass

    @abstractmethod
    def select_device(self, device_id: int = None, device_tag: str = None):
        # Select a measure either by its id, or by its unique tag.
        pass

    @abstractmethod
    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode, gap_tolerance: int = 0):
        # Insert a file path to file index.
        # Insert block_index rows with foreign key file_id.
        # Insert interval_index rows.
        pass

    @abstractmethod
    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int], gap_tolerance: int = 0):
        pass

    @abstractmethod
    def select_file(self, file_id: int = None, file_path: str = None):
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
    def select_block(self, block_id: int = None, measure_id: int = None, device_id: int = None, file_id: int = None,
                     start_byte: int = None, num_bytes: int = None, start_time_n: int = None, end_time_n: int = None,
                     num_values: int = None):
        # select a block either by its id or by all other params.
        pass

    @abstractmethod
    def select_interval(self, interval_id: int = None, measure_id: int = None, device_id: int = None,
                        start_time_n: int = None, end_time_n: int = None):
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
    def insert_patient(self, patient_id=None, mrn: str = None, gender: str = None, dob: str = None,
                       first_name: str = None, middle_name: str = None, last_name: str = None, first_seen: int = None,
                       last_updated: int = None, source_id: int = 1, weight=None, height=None):
        # Insert patient if it doesn't exist, return id.
        pass

    def insert_patient_history(self, patient_id, field, value, units, time):
        with self.connection(begin=True) as (conn, cursor):
            # find the most recent value for the field your entering from the patient history table
            cursor.execute("SELECT MAX(time) FROM patient_history WHERE patient_id = ? and field = ?", (patient_id, field))
            newest_measurement_time = cursor.fetchone()[0]

            # if the new measurement is newer than the newest in the patient history table update the field in the
            # patient table. If no history is found also update it.
            if newest_measurement_time is None or time > newest_measurement_time:
                cursor.execute(f"UPDATE patient SET {field} = {value} WHERE id = {patient_id}")

            # now insert the row to the patient history table
            query = "INSERT INTO patient_history (patient_id, field, value, units, time) VALUES (?, ?, ?, ?, ?)"
            cursor.execute(query, (patient_id, field, value, units, time))
            conn.commit()
            return cursor.lastrowid

    @abstractmethod
    def insert_encounter(self, patient_id: int, bed_id: int, start_time: int, end_time: int = None, source_id: int = 1,
                         visit_number: int = None, last_updated: int = None):
        # Insert encounter if it doesn't exist, return id.
        pass

    @abstractmethod
    def insert_device_encounter(self, device_id: int, encounter_id: int, start_time: int, end_time: int = None,
                                source_id: int = 1):
        # Insert device_encounter if it doesn't exist, return id.
        pass

    def get_device_time_ranges_by_patient(self, patient_id: int, end_time_n: int, start_time_n: int):
        patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ?"
        args = (patient_id,)

        if start_time_n is not None:
            patient_device_query += " AND end_time >= ? "
            args += (start_time_n,)
        if end_time_n is not None:
            patient_device_query += " AND start_time <= ? "
            args += (end_time_n,)
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(patient_device_query, args)
            return cursor.fetchall()
    @abstractmethod
    def select_all_settings(self):
        # Select all settings from settings table.
        pass

    @abstractmethod
    def select_blocks(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        # Get all matching blocks.
        pass

    def select_intervals(self, measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None):
        if device_id is None and patient_id is None:
            raise ValueError("Either device_id or patient_id must be provided")

        interval_query = "SELECT id, measure_id, device_id, start_time_n, end_time_n FROM interval_index WHERE measure_id = ? AND device_id = ?"

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)
            # add start and end time to the query
            interval_query += " AND end_time_n >= ? AND start_time_n <= ? ORDER BY start_time_n ASC, end_time_n ASC"

            interval_results = []
            with self.connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    args = (measure_id, encounter_device_id, encounter_start_time, encounter_end_time)

                    cursor.execute(interval_query, args)
                    interval_results.extend(cursor.fetchall())
            return interval_results

        # Query by device.
        args = (measure_id, device_id)

        if end_time_n is not None:
            interval_query += " AND start_time_n <= ?"
            args += (end_time_n,)

        if start_time_n is not None:
            interval_query += " AND end_time_n >= ?"
            args += (start_time_n,)

        # add the ordering
        interval_query += " ORDER BY start_time_n ASC, end_time_n ASC"

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(interval_query, args)
            return cursor.fetchall()

    @abstractmethod
    def select_encounters(self, patient_id_list: List[int] = None, mrn_list: List[int] = None, start_time: int = None,
                          end_time: int = None):
        # Get all matching encounters.
        pass

    @abstractmethod
    def select_all_measures_in_list(self, measure_id_list: List[int]):
        # Get all matching measures.
        pass

    @abstractmethod
    def select_all_patients_in_list(self, patient_id_list: List[int] = None, mrn_list: List[int] = None):
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

    @abstractmethod
    def select_device_patients(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                               start_time: int = None, end_time: int = None):
        # Get all device_patient rows.
        pass

    @abstractmethod
    def insert_device_patients(self, device_patient_data: List[Tuple[int, int, int, int]]):
        # Insert device_patient rows.
        pass

    def insert_label_set(self, name, label_set_id=None, parent_id=None):
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
        # Retrieve all label types.
        pass

    def select_label_set(self, label_set_id: int):
        query = "SELECT id, name, parent_id FROM label_set WHERE id = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (label_set_id,))
            row = cursor.fetchone()
        return row

    def select_label_set_id(self, name):
        # Retrieve the ID of a label type by its name.
        query = "SELECT id FROM label_set WHERE name = ? LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            # Return the ID if it exists or None otherwise.
            if result:
                return result[0]
            return None

    def select_label_name_parent(self, label_set_id: int):
        query = """
        SELECT parent.id, parent.name FROM label_set
        INNER JOIN label_set AS parent ON label_set.parent_id = parent.id
        WHERE {}
        LIMIT 1;
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query.format("label_set.id = ?"), (label_set_id,))
            return cursor.fetchone()

    def select_all_ancestors(self, label_set_id: int = None, name: str = None):
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
                cursor.execute(query.format("id = ?"), (label_set_id, label_set_id))
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

    def select_label_name_children(self, label_set_id: int):
        query = """
        SELECT id, name FROM label_set
        WHERE parent_id = ?
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query, (label_set_id,))
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
            cursor.execute(query, (label_set_id, label_set_id))
            return cursor.fetchall()

    @abstractmethod
    def insert_label(self, label_set_id, device_id, start_time_n, end_time_n, label_source_id=None):
        # Insert a single label entry and return its ID.
        pass

    @abstractmethod
    def insert_labels(self, labels):
        # Insert multiple label entries and return their IDs.
        # `labels` is a list of tuples (label_set_id, device_id, start_time_n, end_time_n, label_source_id)
        pass

    @abstractmethod
    def delete_labels(self, label_ids):
        # Delete multiple label records from the database based on their IDs.
        pass

    @abstractmethod
    def select_labels(self, label_set_id_list=None, device_id_list=None, patient_id_list=None, start_time_n=None,
                      end_time_n=None, label_source_id_list=None):
        # Retrieve labels based on provided criteria.
        pass

    @abstractmethod
    def insert_label_source(self, name, description):
        pass

    @abstractmethod
    def select_label_source_id_by_name(self, name):
        pass

    @abstractmethod
    def select_label_source_info_by_id(self, label_source_id):
        pass

    def get_measure_id_with_most_rows(self, tag: str):
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

    def get_tag_to_measure_ids_dict(self, approx=True):
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
                id_to_tag[tag].append(measure_id)

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

        measure_id_to_count = {}
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_index_query)
            for row in cursor.fetchall():
                measure_id, count = row
                measure_id_to_count[measure_id] = count

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

    def insert_source(self, name: str, description: str = None):
        query = "INSERT INTO source (name, description) VALUES (?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name, description))
            conn.commit()
            return cursor.lastrowid

    def select_source(self, source_id: int = None, name: str = None):
        query = "SELECT id, name, description FROM source WHERE "
        params = []
        if source_id:
            query += "id = ?"
            params.append(source_id)
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either source_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_institution(self, name: str):
        query = "INSERT INTO institution (name) VALUES (?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name,))
            conn.commit()
            return cursor.lastrowid

    def select_institution(self, institution_id: int = None, name: str = None):
        query = "SELECT id, name FROM institution WHERE "
        params = []
        if institution_id:
            query += "id = ?"
            params.append(institution_id)
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either institution_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_unit(self, institution_id: int, name: str, unit_type: str):
        query = "INSERT INTO unit (institution_id, name, type) VALUES (?, ?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (institution_id, name, unit_type))
            conn.commit()
            return cursor.lastrowid

    def select_unit(self, unit_id: int = None, name: str = None):
        query = "SELECT id, institution_id, name, type FROM unit WHERE "
        params = []
        if unit_id:
            query += "id = ?"
            params.append(unit_id)
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either unit_id or name must be provided")

        with self.connection() as (conn, cursor):
            
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_bed(self, unit_id: int, name: str):
        query = "INSERT INTO bed (unit_id, name) VALUES (?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (unit_id, name))
            conn.commit()
            return cursor.lastrowid

    def select_bed(self, bed_id: int = None, name: str = None):
        query = "SELECT id, unit_id, name FROM bed WHERE "
        params = []
        if bed_id:
            query += "id = ?"
            params.append(bed_id)
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either bed_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()
