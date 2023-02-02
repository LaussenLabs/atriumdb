maria_measures_table_create_query = """CREATE TABLE IF NOT EXISTS measures(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_tag VARCHAR(64) NOT NULL,
    measure_name VARCHAR(190),
    freq_nhz BIGINT NOT NULL,
    units VARCHAR(64),
    UNIQUE KEY (measure_tag, freq_nhz, units)
);"""

maria_devices_table_create_query = """CREATE TABLE IF NOT EXISTS devices(
    id INT PRIMARY KEY AUTO_INCREMENT,
    device_tag VARCHAR(64) NOT NULL,
    device_name VARCHAR(190),
    UNIQUE KEY (device_tag)
);"""

maria_file_index_create_query = """CREATE TABLE IF NOT EXISTS file_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT NOT NULL,
    device_id INT NOT NULL,
    path VARCHAR(190) NOT NULL,
    UNIQUE KEY (path),
    INDEX (measure_id, device_id)
);"""

maria_block_index_create_query = """CREATE TABLE IF NOT EXISTS block_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT NOT NULL,
    device_id INT NOT NULL,
    file_id INT NOT NULL,
    start_byte BIGINT NOT NULL,
    num_bytes BIGINT NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    num_values BIGINT NOT NULL,
    INDEX (measure_id, device_id, start_time_n, end_time_n)
);"""

maria_interval_index_create_query = """CREATE TABLE IF NOT EXISTS interval_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT NOT NULL,
    device_id INT NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    INDEX (measure_id, device_id, start_time_n, end_time_n)
);"""

maria_encounter_create_query = """CREATE TABLE IF NOT EXISTS encounter(
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    device_id INT NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    INDEX (patient_id, device_id, start_time_n, end_time_n)
);"""

maria_settings_create_query = """CREATE TABLE IF NOT EXISTS settings(
    setting_name VARCHAR(64) PRIMARY KEY,
    setting_value VARCHAR(64) NOT NULL
);"""

