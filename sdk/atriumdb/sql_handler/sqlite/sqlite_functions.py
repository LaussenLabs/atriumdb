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

sqlite_insert_ignore_measure_query = "INSERT OR IGNORE INTO measure (tag, freq_nhz, unit, name) " \
                              "VALUES (?, ?, ?, ?);"
sqlite_select_measure_from_id_query = "SELECT id, tag, name, freq_nhz, code, unit, unit_label, unit_code, source_id FROM measure WHERE id = ?"
sqlite_select_measure_from_triplet_query = "SELECT id, tag, name, freq_nhz, code, unit, unit_label, unit_code, source_id FROM measure WHERE tag = ? AND freq_nhz = ? AND unit = ?"

sqlite_insert_ignore_device_query = "INSERT OR IGNORE INTO device (tag, name) VALUES (?, ?);"
sqlite_select_device_from_id_query = "SELECT id, tag, name, manufacturer, model, type, bed_id, source_id FROM device WHERE id = ?"
sqlite_select_device_from_tag_query = "SELECT id, tag, name, manufacturer, model, type, bed_id, source_id FROM device WHERE tag = ?"

sqlite_insert_block_query = """INSERT INTO block_index 
(measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""

sqlite_insert_file_index_query = """INSERT INTO file_index (path) VALUES (?);"""

sqlite_insert_interval_index_query = """INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
VALUES (?, ?, ?, ?);"""

sqlite_select_block_by_id = "SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values FROM block_index WHERE id = ?;"
sqlite_select_block_by_values = """SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values FROM block_index 
WHERE measure_id = ? AND device_id = ? AND file_id = ? AND start_byte = ? AND num_bytes = ? AND start_time_n = ? AND end_time_n = ? AND num_values = ?;"""

sqlite_select_blocks_from_file = "SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values FROM block_index WHERE file_id = ? ORDER BY start_byte;"
sqlite_delete_block_query = "DELETE FROM block_index WHERE id = ?;"

sqlite_select_file_by_id = "SELECT id, path FROM file_index WHERE id = ?;"
sqlite_select_file_by_values = """SELECT id, path FROM file_index WHERE path = ?;
"""
sqlite_delete_file_query = "DELETE FROM file_index WHERE id = ?;"

sqlite_select_interval_by_id = "SELECT id, measure_id, device_id, start_time_n, end_time_n FROM interval_index WHERE id = ?;"
sqlite_select_interval_by_values = """SELECT id, measure_id, device_id, start_time_n, end_time_n FROM interval_index 
WHERE measure_id = ? AND device_id = ? AND start_time_n = ? AND end_time_n = ?;
"""
sqlite_interval_exists_query = "SELECT EXISTS(SELECT id FROM interval_index WHERE measure_id = ? AND device_id = ? AND start_time_n = ?)"


sqlite_setting_insert_query = "INSERT OR IGNORE INTO setting (name, value) VALUES (?, ?)"
sqlite_setting_select_query = "SELECT name, value FROM setting WHERE name = ?"
sqlite_setting_select_all_query = "SELECT name, value FROM setting"

sqlite_insert_ignore_source_query = "INSERT OR IGNORE INTO source (name, description) VALUES (?, ?);"
sqlite_insert_ignore_institution_query = "INSERT OR IGNORE INTO institution (name) VALUES (?);"
sqlite_insert_ignore_unit_query = "INSERT OR IGNORE INTO unit (institution_id, name, type) VALUES (?, ?, ?);"
sqlite_insert_ignore_bed_query = "INSERT OR IGNORE INTO bed (unit_id, name) VALUES (?, ?);"
sqlite_insert_ignore_patient_query = "INSERT OR IGNORE INTO patient (id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id, weight, height) " \
                                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
sqlite_insert_ignore_encounter_query = "INSERT OR IGNORE INTO encounter (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated) " \
                                       "VALUES (?, ?, ?, ?, ?, ?, ?);"
sqlite_insert_ignore_device_encounter_query = "INSERT OR IGNORE INTO device_encounter (device_id, encounter_id, start_time, end_time, source_id) " \
                                              "VALUES (?, ?, ?, ?, ?);"


def sqlite_get_query_with_patient_id(measure_id, patient_id, start_time_n=None, end_time_n=None):
    query = \
        "FROM block_index " \
        "INNER JOIN device_patient ON device_patient.device_id = block_index.device_id " \
        "WHERE block_index.measure_id = ? " \
        "AND device_patient.patient_id = ? " \
        "AND device_patient.start_time < block_index.end_time_n " \
        "AND block_index.start_time_n < device_patient.end_time "
    arg_tuple = (measure_id, patient_id)
    if start_time_n is not None:
        query += " AND block_index.end_time_n >= ?"
        arg_tuple += (start_time_n,)
    if end_time_n is not None:
        query += " AND block_index.start_time_n <= ?"
        arg_tuple += (end_time_n,)
    return query, arg_tuple
