sqlite_measures_table_create_query = """CREATE TABLE IF NOT EXISTS measures(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_tag TEXT NOT NULL,
    measure_name TEXT,
    freq_nhz INTEGER NOT NULL,
    units TEXT,
    UNIQUE (measure_tag, freq_nhz, units)
);"""

sqlite_devices_table_create_query = """CREATE TABLE IF NOT EXISTS devices(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_tag TEXT NOT NULL,
    device_name TEXT,
    UNIQUE (device_tag)
);"""

sqlite_file_index_create_query = """CREATE TABLE IF NOT EXISTS file_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    UNIQUE (path)
);"""

sqlite_file_index_idx_query = "CREATE INDEX IF NOT EXISTS file_idx ON file_index (measure_id, device_id);"

sqlite_block_index_create_query = """CREATE TABLE IF NOT EXISTS block_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    num_bytes INTEGER NOT NULL,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL,
    num_values INTEGER NOT NULL
);"""

sqlite_block_index_idx_query = \
    "CREATE INDEX IF NOT EXISTS block_idx ON block_index (measure_id, device_id, start_time_n, end_time_n);"

sqlite_interval_index_create_query = """CREATE TABLE IF NOT EXISTS interval_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL
);"""

sqlite_interval_index_idx_query = \
    "CREATE INDEX IF NOT EXISTS interval_idx ON interval_index (measure_id, device_id, start_time_n, end_time_n);"

sqlite_encounter_create_query = """CREATE TABLE IF NOT EXISTS encounter(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL
);"""

sqlite_encounter_index_idx_query = \
    "CREATE INDEX IF NOT EXISTS encounter_idx ON encounter (patient_id, device_id, start_time_n, end_time_n);"

sqlite_settings_create_query = """CREATE TABLE IF NOT EXISTS settings(
    setting_name TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL
);"""
