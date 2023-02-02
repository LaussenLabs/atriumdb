maria_insert_ignore_measure_query = "INSERT IGNORE INTO measures (measure_tag, freq_nhz, units, measure_name) " \
                              "VALUES (?, ?, ?, ?);"
maria_select_measure_from_id_query = "SELECT * FROM measures WHERE id = ?"
maria_select_measure_from_triplet_query = "SELECT * FROM measures WHERE measure_tag = ? AND freq_nhz = ? AND units = ?"

maria_insert_ignore_device_query = "INSERT IGNORE INTO devices (device_tag, device_name) VALUES (?, ?);"
maria_select_device_from_id_query = "SELECT * FROM devices WHERE id = ?"
maria_select_device_from_tag_query = "SELECT * FROM devices WHERE device_tag = ?"