mariadb_measure_create_query = """
CREATE TABLE IF NOT EXISTS measure (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
tag VARCHAR(64) NOT NULL,
name VARCHAR(255) NULL,
freq_nhz BIGINT NOT NULL,
code VARCHAR(64) NULL,
unit VARCHAR(64) NOT NULL,
unit_label VARCHAR(255) NULL,
unit_code VARCHAR(64) NULL,
source_id INT UNSIGNED DEFAULT 1 NULL,
CONSTRAINT source_ibfk_1 FOREIGN KEY (source_id) REFERENCES source (id),
CONSTRAINT tag_frequency_unit UNIQUE (tag, freq_nhz, unit)
);
"""

mariadb_measure_source_id_create_index = """
CREATE INDEX IF NOT EXISTS source_id ON measure (source_id);
"""

maria_file_index_create_query = """CREATE TABLE IF NOT EXISTS file_index(
    id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    path VARCHAR(255) NOT NULL,
    UNIQUE KEY (path)
);"""

maria_block_index_create_query = """CREATE TABLE IF NOT EXISTS block_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT UNSIGNED NOT NULL,
    device_id INT UNSIGNED NOT NULL,
    file_id INT UNSIGNED NOT NULL,
    start_byte BIGINT NOT NULL,
    num_bytes BIGINT NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    num_values BIGINT NOT NULL,
    CONSTRAINT block_index_ibfk_1 FOREIGN KEY (measure_id) REFERENCES measure (id),
    CONSTRAINT block_index_ibfk_2 FOREIGN KEY (device_id) REFERENCES device (id),
    CONSTRAINT block_index_ibfk_3 FOREIGN KEY (file_id) REFERENCES file_index (id),
    INDEX (start_time_n, end_time_n)
);"""

maria_interval_index_create_query = """CREATE TABLE IF NOT EXISTS interval_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT UNSIGNED NOT NULL,
    device_id INT UNSIGNED NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    CONSTRAINT INTerval_index_ibfk_1 FOREIGN KEY (measure_id) REFERENCES measure(id),
    CONSTRAINT INTerval_index_ibfk_2 FOREIGN KEY (device_id) REFERENCES device(id),
    INDEX (start_time_n, end_time_n)
);"""


maria_settings_create_query = """CREATE TABLE IF NOT EXISTS setting(
    name VARCHAR(64) PRIMARY KEY,
    value VARCHAR(64) NOT NULL
);"""

maria_source_create_query = """
CREATE TABLE IF NOT EXISTS source (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description VARCHAR(255) NULL
);
"""

maria_insert_adb_source = 'INSERT IGNORE INTO source (id, name, description) VALUES (1, "AtriumDB", "AtriumDB System");'

maria_institution_create_query = """
CREATE TABLE IF NOT EXISTS institution (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL
);
"""

maria_unit_create_query = """
CREATE TABLE IF NOT EXISTS unit (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  institution_id INT UNSIGNED NOT NULL,
  name VARCHAR(255) NOT NULL,
  type VARCHAR(255) NOT NULL,
  CONSTRAINT unit_ibfk_1 FOREIGN KEY (institution_id) REFERENCES institution (id)
);
"""

maria_unit_institution_id_create_index = "CREATE INDEX IF NOT EXISTS institutionId ON unit (institution_id);"

maria_bed_create_query = """
CREATE TABLE IF NOT EXISTS bed (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  unit_id INT UNSIGNED NOT NULL,
  name VARCHAR(255) NOT NULL,
  CONSTRAINT bed_ibfk_1 FOREIGN KEY (unit_id) REFERENCES unit (id)
);
"""

maria_bed_unit_id_create_index = "CREATE INDEX IF NOT EXISTS unit_id ON bed (unit_id);"

maria_patient_create_query = """
CREATE TABLE IF NOT EXISTS patient (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  mrn INT UNSIGNED NOT NULL,
  gender VARCHAR(1) NULL,
  dob bigint NULL,
  first_name VARCHAR(255) NULL,
  middle_name VARCHAR(255) NULL,
  last_name VARCHAR(255) NULL,
  first_seen bigint DEFAULT CURRENT_TIMESTAMP NOT NULL,
  last_updated bigint NULL,
  source_id INT UNSIGNED DEFAULT 1 NULL,
  CONSTRAINT mrn UNIQUE (mrn),
  CONSTRAINT patient_ibfk_1 FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

maria_patient_index_query = """
CREATE INDEX IF NOT EXISTS source_id ON patient (source_id);
"""

maria_encounter_create_query = """
CREATE TABLE IF NOT EXISTS encounter (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  patient_id INT UNSIGNED null,
  bed_id INT UNSIGNED null,
  start_time bigint not null,
  end_time bigint null,
  source_id INT UNSIGNED DEFAULT 1 null,
  visit_number VARCHAR(15) null,
  last_updated bigint not null,
  CONSTRAINT encounter_ibfk_1 FOREIGN KEY (patient_id) REFERENCES patient (id),
  CONSTRAINT encounter_ibfk_2 FOREIGN KEY (bed_id) REFERENCES bed (id),
  CONSTRAINT encounter_ibfk_3 FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

maria_encounter_create_index_bed_id_query = """
CREATE INDEX IF NOT EXISTS bed_id ON encounter (bed_id);
"""

maria_encounter_create_index_patient_id_query = """
CREATE INDEX IF NOT EXISTS patient_id ON encounter (patient_id);
"""

maria_encounter_create_index_source_id_query = """
CREATE INDEX IF NOT EXISTS source_id ON encounter (source_id);
"""

mariadb_device_create_query = """
CREATE TABLE IF NOT EXISTS device (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
tag VARCHAR(64) NOT NULL,
name VARCHAR(255) NULL,
manufacturer VARCHAR(255) NULL,
model VARCHAR(255) NULL,
type ENUM('static', 'dynamic') DEFAULT 'static' NOT NULL,
bed_id INT UNSIGNED NULL,
source_id INT UNSIGNED DEFAULT 1 NOT NULL,
UNIQUE KEY (tag),
CONSTRAINT device_ibfk_2 FOREIGN KEY (source_id) REFERENCES source (id),
CONSTRAINT device_ibfk_3 FOREIGN KEY (bed_id) REFERENCES bed (id)
);
"""

mariadb_device_bed_id_create_index = """
CREATE INDEX IF NOT EXISTS bed_id ON device (bed_id);
"""

mariadb_device_source_id_create_index = """
CREATE INDEX IF NOT EXISTS source_id ON device (source_id);
"""


maria_device_encounter_create_query = """CREATE TABLE IF NOT EXISTS device_encounter (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  device_id INT UNSIGNED NOT NULL,
  encounter_id INT UNSIGNED NOT NULL,
  start_time BIGINT NOT NULL,
  end_time BIGINT NULL,
  source_id INT UNSIGNED DEFAULT 1 NULL,
  CONSTRAINT device_encounter_ibfk_1 FOREIGN KEY (device_id) REFERENCES device (id) ON DELETE CASCADE,
  CONSTRAINT device_encounter_ibfk_2 FOREIGN KEY (encounter_id) REFERENCES encounter (id) ON DELETE CASCADE,
  CONSTRAINT device_encounter_ibfk_3 FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

maria_device_encounter_device_id_create_index = "CREATE INDEX IF NOT EXISTS device_id_index ON device_encounter (device_id);"
maria_device_encounter_encounter_id_create_index = "CREATE INDEX IF NOT EXISTS encounter_id_index ON device_encounter (encounter_id);"
maria_device_encounter_source_id_create_index = "CREATE INDEX IF NOT EXISTS source_id_index ON device_encounter (source_id);"


mariadb_log_hl7_adt_create_query = """CREATE TABLE IF NOT EXISTS log_hl7_adt(
id INT PRIMARY KEY AUTO_INCREMENT,
event_type VARCHAR(15) NOT NULL,
event_time BIGINT NOT NULL,
mrn VARCHAR(255) DEFAULT NULL,
first_name VARCHAR(255) DEFAULT NULL,
middle_name VARCHAR(255) DEFAULT NULL,
last_name VARCHAR(255) DEFAULT NULL,
gender VARCHAR(3) DEFAULT NULL,
dob BIGINT DEFAULT NULL,
visit_num VARCHAR(255) DEFAULT NULL,
visit_class VARCHAR(255) DEFAULT NULL,
location VARCHAR(255) DEFAULT NULL,
previous_location VARCHAR(255) DEFAULT NULL,
admit_time BIGINT DEFAULT NULL,
discharge_time BIGINT DEFAULT NULL,
source_id INT UNSIGNED DEFAULT 1 NULL,
CONSTRAINT log_hl7_adt_ibfk_1 FOREIGN KEY (source_id) REFERENCES source (id)
);"""

mariadb_log_hl7_adt_source_id_create_index = """CREATE INDEX IF NOT EXISTS source_id
ON log_hl7_adt (source_id);"""

mariadb_current_census_view = """CREATE OR REPLACE VIEW current_census AS select `e`.`start_time` AS 
`admission_start`,`u`.`id` AS `unit_id`,`u`.`name` AS `unit_name`,`b`.`id` AS `bed_id`,`b`.`name` AS `bed_name`,
`p`.`id` AS `patient_id`,`p`.`mrn` AS `mrn`,concat_ws(' ',`p`.`first_name`,`p`.`middle_name`,`p`.`last_name`) AS `name`,
`p`.`gender` AS `gender`,`p`.`dob` AS `birth_date` from (((`encounter` `e` join `bed` `b` on(`e`.`bed_id` = `b`.`id`)) 
join `unit` `u` on(`b`.`unit_id` = `u`.`id`)) join `patient` `p` on(`e`.`patient_id` = `p`.`id`)) 
where `e`.`end_time` is null
"""
