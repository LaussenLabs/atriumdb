sqlite_insert_ignore_measure_query = "INSERT OR IGNORE INTO measure (tag, freq_nhz, unit, name) " \
                              "VALUES (?, ?, ?, ?);"
sqlite_select_measure_from_id_query = "SELECT * FROM measure WHERE id = ?"
sqlite_select_measure_from_triplet_query = "SELECT * FROM measure WHERE tag = ? AND freq_nhz = ? AND unit = ?"

sqlite_insert_ignore_device_query = "INSERT OR IGNORE INTO device (tag, name) VALUES (?, ?);"
sqlite_select_device_from_id_query = "SELECT * FROM device WHERE id = ?"
sqlite_select_device_from_tag_query = "SELECT * FROM device WHERE tag = ?"

sqlite_insert_block_query = """INSERT INTO block_index 
(measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""

sqlite_insert_file_index_query = """INSERT INTO file_index (path) VALUES (?);"""

sqlite_insert_interval_index_query = """INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
VALUES (?, ?, ?, ?);"""

sqlite_select_block_by_id = "SELECT * FROM block_index WHERE id = ?;"
sqlite_select_block_by_values = """SELECT * FROM block_index 
WHERE measure_id = ? AND device_id = ? AND file_id = ? 
AND start_byte = ? AND num_bytes = ? AND start_time_n = ? 
AND end_time_n = ? AND num_values = ?;
"""
sqlite_select_blocks_from_file = "SELECT * FROM block_index WHERE file_id = ? ORDER BY start_byte;"
sqlite_delete_block_query = "DELETE FROM block_index WHERE id = ?;"

sqlite_select_file_by_id = "SELECT * FROM file_index WHERE id = ?;"
sqlite_select_file_by_values = """SELECT * FROM file_index WHERE path = ?;
"""
sqlite_delete_file_query = "DELETE FROM file_index WHERE id = ?;"

sqlite_select_interval_by_id = "SELECT * FROM interval_index WHERE id = ?;"
sqlite_select_interval_by_values = """SELECT * FROM interval_index 
WHERE measure_id = ? AND device_id = ? AND start_time_n = ? AND end_time_n = ?;
"""
sqlite_interval_exists_query = """
    SELECT EXISTS(
      SELECT id
      FROM interval_index
      WHERE measure_id = ?
        AND device_id = ?
        AND start_time_n = ?
    )
    """


sqlite_setting_insert_query = "INSERT OR IGNORE INTO setting (name, value) VALUES (?, ?)"
sqlite_setting_select_query = "SELECT * FROM setting WHERE name = ?"
sqlite_setting_select_all_query = "SELECT * FROM setting"

sqlite_insert_ignore_source_query = "INSERT OR IGNORE INTO source (name, description) VALUES (?, ?);"
sqlite_insert_ignore_institution_query = "INSERT OR IGNORE INTO institution (name) VALUES (?);"
sqlite_insert_ignore_unit_query = "INSERT OR IGNORE INTO unit (institution_id, name, type) VALUES (?, ?, ?);"
sqlite_insert_ignore_bed_query = "INSERT OR IGNORE INTO bed (unit_id, name) VALUES (?, ?);"
sqlite_insert_ignore_patient_query = "INSERT OR IGNORE INTO patient (id, mrn, gender, dob, first_name, middle_name, last_name, first_seen, last_updated, source_id) " \
                                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
sqlite_insert_ignore_encounter_query = "INSERT OR IGNORE INTO encounter (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated) " \
                                       "VALUES (?, ?, ?, ?, ?, ?, ?);"
sqlite_insert_ignore_device_encounter_query = "INSERT OR IGNORE INTO device_encounter (device_id, encounter_id, start_time, end_time, source_id) " \
                                              "VALUES (?, ?, ?, ?, ?);"