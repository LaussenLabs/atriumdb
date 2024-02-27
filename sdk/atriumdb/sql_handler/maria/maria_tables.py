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
CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id),
CONSTRAINT tag_frequency_unit UNIQUE (tag, freq_nhz, unit)
);
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
    CONSTRAINT FOREIGN KEY (measure_id) REFERENCES measure (id),
    CONSTRAINT FOREIGN KEY (device_id) REFERENCES device (id),
    CONSTRAINT FOREIGN KEY (file_id) REFERENCES file_index (id)
    ON DELETE CASCADE,
    INDEX (measure_id, device_id, start_time_n, end_time_n)
);"""

maria_interval_index_create_query = """CREATE TABLE IF NOT EXISTS interval_index(
    id INT PRIMARY KEY AUTO_INCREMENT,
    measure_id INT UNSIGNED NOT NULL,
    device_id INT UNSIGNED NOT NULL,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    CONSTRAINT FOREIGN KEY (measure_id) REFERENCES measure(id),
    CONSTRAINT FOREIGN KEY (device_id) REFERENCES device(id),
    INDEX (measure_id ASC, device_id ASC, end_time_n DESC, start_time_n DESC)
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
  CONSTRAINT FOREIGN KEY (institution_id) REFERENCES institution (id)
);
"""

maria_bed_create_query = """
CREATE TABLE IF NOT EXISTS bed (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  unit_id INT UNSIGNED NOT NULL,
  name VARCHAR(255) NOT NULL,
  CONSTRAINT FOREIGN KEY (unit_id) REFERENCES unit (id)
);
"""

maria_patient_create_query = """
CREATE TABLE IF NOT EXISTS patient (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  mrn INT UNSIGNED NULL,
  gender VARCHAR(1) NULL,
  dob bigint NULL,
  first_name VARCHAR(255) NULL,
  middle_name VARCHAR(255) NULL,
  last_name VARCHAR(255) NULL,
  first_seen bigint DEFAULT CURRENT_TIMESTAMP NULL,
  last_updated bigint NULL,
  source_id INT UNSIGNED DEFAULT 1 NULL,
  weight FLOAT UNSIGNED NULL,
  height FLOAT UNSIGNED NULL,
  CONSTRAINT mrn UNIQUE (mrn),
  CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

maria_patient_history_create_query = """
CREATE TABLE IF NOT EXISTS patient_history (
    id INT UNSIGNED auto_increment PRIMARY KEY,
    patient_id INT UNSIGNED NOT NULL,
    field VARCHAR(120) NOT NULL,
    value FLOAT UNSIGNED NOT NULL,
    units VARCHAR(64) NULL,
    time BIGINT NOT NULL,
    CONSTRAINT FOREIGN KEY (patient_id) REFERENCES patient (id) ON DELETE CASCADE,
    INDEX (patient_id, field, time DESC)
);
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
  CONSTRAINT FOREIGN KEY (patient_id) REFERENCES patient (id),
  CONSTRAINT FOREIGN KEY (bed_id) REFERENCES bed (id),
  CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id)
);
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
CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id),
CONSTRAINT FOREIGN KEY (bed_id) REFERENCES bed (id)
);
"""

maria_device_encounter_create_query = """CREATE TABLE IF NOT EXISTS device_encounter (
  id INT UNSIGNED auto_increment PRIMARY KEY,
  device_id INT UNSIGNED NOT NULL,
  encounter_id INT UNSIGNED NOT NULL,
  start_time BIGINT NOT NULL,
  end_time BIGINT NULL,
  source_id INT UNSIGNED DEFAULT 1 NULL,
  CONSTRAINT FOREIGN KEY (device_id) REFERENCES device (id) ON DELETE CASCADE,
  CONSTRAINT FOREIGN KEY (encounter_id) REFERENCES encounter (id) ON DELETE CASCADE,
  CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

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
weight FLOAT DEFAULT NULL,
height FLOAT DEFAULT NULL,
CONSTRAINT FOREIGN KEY (source_id) REFERENCES source (id)
);"""

mariadb_current_census_view = """CREATE OR REPLACE VIEW current_census AS select `e`.`start_time` AS `admission_start`,
`u`.`id` AS `unit_id`,`u`.`name` AS `unit_name`,`b`.`id` AS `bed_id`,`b`.`name` AS `bed_name`,`p`.`id` AS `patient_id`,
`p`.`mrn` AS `mrn`,`p`.`first_name` AS `first_name`,`p`.`middle_name` AS `middle_name`,`p`.`last_name` AS `last_name`,`p`.`gender` AS `gender`,
`p`.`dob` AS `birth_date`,`p`.`weight` AS `weight`,`p`.`height` AS `height` from (((`encounter` `e` join `bed` `b` 
on(`e`.`bed_id` = `b`.`id`)) join `unit` `u` on(`b`.`unit_id` = `u`.`id`)) join `patient` `p` on(`e`.`patient_id` = `p`.`id`)) 
where `e`.`end_time` is null
"""

mariadb_device_patient_table = """CREATE TABLE IF NOT EXISTS device_patient (
id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
device_id INT UNSIGNED NOT NULL,
patient_id INT UNSIGNED NOT NULL,
start_time BIGINT NOT NULL,
end_time BIGINT NULL,
source_id INT UNSIGNED NOT NULL DEFAULT 1,
FOREIGN KEY (device_id) REFERENCES device(id) ON DELETE CASCADE,
FOREIGN KEY (patient_id) REFERENCES patient(id) ON DELETE CASCADE,
FOREIGN KEY (source_id) REFERENCES source(id),
INDEX (device_id, patient_id, start_time, end_time)
);
"""

maria_encounter_insert_trigger = """
CREATE TRIGGER IF NOT EXISTS encounter_insert
    after insert
    on encounter
    for each row
BEGIN
INSERT INTO device_encounter (device_id, encounter_id, start_time, end_time) 
SELECT id, NEW.id, NEW.start_time, NEW.end_time 
FROM device WHERE type='static' and bed_id=NEW.bed_id;

INSERT INTO device_patient (device_id, patient_id, start_time, end_time, source_id) 
SELECT id, NEW.patient_id, NEW.start_time, NEW.end_time, NEW.source_id 
FROM device WHERE type='static' and bed_id=NEW.bed_id;
END;
"""

maria_encounter_update_trigger = """
CREATE TRIGGER IF NOT EXISTS encounter_update
    after update
    on encounter
    for each row
BEGIN
UPDATE device_encounter SET end_time=NEW.end_time WHERE encounter_id=NEW.id and end_time IS NULL;
UPDATE device_patient SET end_time=NEW.end_time WHERE patient_id=NEW.patient_id AND start_time=NEW.start_time AND end_time IS NULL;
END;
"""

maria_encounter_delete_trigger="""
CREATE TRIGGER IF NOT EXISTS encounter_delete
    after delete
    on encounter
    for each row
BEGIN
DELETE FROM device_patient WHERE patient_id=OLD.patient_id AND start_time=OLD.start_time;
END;
"""


maria_insert_interval_stored_procedure = """
CREATE PROCEDURE IF NOT EXISTS insert_interval(
    IN p_measure_id INT UNSIGNED,
    IN p_device_id INT UNSIGNED,
    IN p_start_time_n BIGINT,
    IN p_end_time_n BIGINT,
    IN p_tolerance BIGINT
)
BEGIN
    DECLARE overlapping_row_id INT;
    DECLARE existing_start_time_n BIGINT;
    DECLARE existing_end_time_n BIGINT;
    DECLARE max_end_time_n BIGINT;

    SELECT MAX(end_time_n) INTO max_end_time_n FROM interval_index WHERE device_id = p_device_id AND measure_id = p_measure_id;

    -- If new start time is greater than the max end time there will be no overlapping interval match so don't check
    IF p_start_time_n > (max_end_time_n + p_tolerance) THEN
        -- Insert the new row into the table
        INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
        VALUES (p_measure_id, p_device_id, p_start_time_n, p_end_time_n);
    ELSE
        -- Check if there's an existing row with overlapping intervals
        SELECT id, start_time_n, end_time_n INTO overlapping_row_id, existing_start_time_n, existing_end_time_n
        FROM interval_index
        WHERE device_id = p_device_id AND measure_id = p_measure_id
        AND p_start_time_n <= (end_time_n + p_tolerance) AND p_end_time_n >= (start_time_n - p_tolerance)
        LIMIT 1;
    
        IF overlapping_row_id IS NOT NULL THEN
            -- Make sure the new end time is greater than the old before updating
            IF p_end_time_n > existing_end_time_n THEN
                -- Update the existing row's end_time_n with the new end_time_n
                UPDATE interval_index
                SET end_time_n = p_end_time_n
                WHERE id = overlapping_row_id;
            END IF;
    
            IF p_start_time_n < existing_start_time_n THEN
                -- Update the existing row's start_time_n with the new start_time_n
                UPDATE interval_index
                SET start_time_n = p_start_time_n
                WHERE id = overlapping_row_id;
            END IF;
        ELSE
            -- Insert the new row into the table
            INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
            VALUES (p_measure_id, p_device_id, p_start_time_n, p_end_time_n);
        END IF;
    END IF;
END;
"""

mariadb_label_set_create_query = """
CREATE TABLE IF NOT EXISTS label_set (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(128) NOT NULL UNIQUE,
parent_id INT UNSIGNED,
FOREIGN KEY (parent_id) REFERENCES label_set(id)
);
"""

mariadb_label_source_create_query = """
CREATE TABLE IF NOT EXISTS label_source (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT
);
"""

mariadb_label_create_query = """
CREATE TABLE IF NOT EXISTS label (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    label_set_id INT UNSIGNED NOT NULL,
    device_id INT UNSIGNED NOT NULL,
    label_source_id INT UNSIGNED,
    start_time_n BIGINT NOT NULL,
    end_time_n BIGINT NOT NULL,
    CONSTRAINT FOREIGN KEY (label_set_id) REFERENCES label_set(id),
    CONSTRAINT FOREIGN KEY (device_id) REFERENCES device(id),
    CONSTRAINT FOREIGN KEY (label_source_id) REFERENCES label_source(id),
    INDEX (label_set_id, device_id, start_time_n, end_time_n),
    INDEX (label_set_id, device_id, label_source_id, start_time_n, end_time_n)
);
"""
