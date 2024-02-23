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

sqlite_measure_create_query = """
CREATE TABLE IF NOT EXISTS measure (
id INTEGER PRIMARY KEY AUTOINCREMENT,
tag TEXT NOT NULL,
name TEXT NULL,
freq_nhz INTEGER NOT NULL,
code TEXT NULL,
unit TEXT NOT NULL,
unit_label TEXT NULL,
unit_code TEXT NULL,
source_id INTEGER DEFAULT 1 NULL,
FOREIGN KEY (source_id) REFERENCES source (id),
UNIQUE (tag, freq_nhz, unit)
);
"""

sqlite_measure_source_id_create_index = """
CREATE INDEX IF NOT EXISTS source_id_index ON measure (source_id);
"""

sqlite_file_index_create_query = """CREATE TABLE IF NOT EXISTS file_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    UNIQUE (path)
);"""

sqlite_block_index_create_query = """CREATE TABLE IF NOT EXISTS block_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    num_bytes INTEGER NOT NULL,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL,
    num_values INTEGER NOT NULL,
    FOREIGN KEY (measure_id) REFERENCES measure(id),
    FOREIGN KEY (device_id) REFERENCES device(id),
    FOREIGN KEY (file_id) REFERENCES file_index(id) ON DELETE CASCADE 
);"""

sqlite_block_index_idx_query = \
    "CREATE INDEX IF NOT EXISTS block_idx ON block_index (measure_id, device_id, start_time_n, end_time_n);"


sqlite_interval_index_create_query = """CREATE TABLE IF NOT EXISTS interval_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measure_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL,
    FOREIGN KEY (measure_id) REFERENCES measure(id),
    FOREIGN KEY (device_id) REFERENCES device(id)
);"""

sqlite_interval_index_idx_query = \
    "CREATE INDEX IF NOT EXISTS interval_idx ON interval_index (measure_id, device_id, start_time_n, end_time_n);"

sqlite_settings_create_query = """CREATE TABLE IF NOT EXISTS setting(
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);"""

sqlite_device_encounter_create_query = """CREATE TABLE IF NOT EXISTS device_encounter (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER NOT NULL,
  encounter_id INTEGER NOT NULL,
  start_time INTEGER NOT NULL,
  end_time INTEGER,
  source_id INTEGER DEFAULT 1,
  FOREIGN KEY (device_id) REFERENCES device (id) ON DELETE CASCADE,
  FOREIGN KEY (encounter_id) REFERENCES encounter (id) ON DELETE CASCADE,
  FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

sqlite_device_encounter_device_id_create_index = "CREATE INDEX IF NOT EXISTS device_id_index ON device_encounter (device_id);"
sqlite_device_encounter_encounter_id_create_index = "CREATE INDEX IF NOT EXISTS encounter_id_index ON device_encounter (encounter_id);"
sqlite_device_encounter_source_id_create_index = "CREATE INDEX IF NOT EXISTS source_id_index ON device_encounter (source_id);"

sqlite_source_create_query = """
CREATE TABLE IF NOT EXISTS source (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT
);
"""

sqlite_insert_adb_source = 'INSERT OR IGNORE INTO source (id, name, description) VALUES (1, "AtriumDB", "AtriumDB System");'

sqlite_institution_create_query = """
CREATE TABLE IF NOT EXISTS institution (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL
);
"""

sqlite_unit_create_query = """
CREATE TABLE IF NOT EXISTS unit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  institution_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  FOREIGN KEY (institution_id) REFERENCES institution (id)
);
"""

sqlite_unit_institution_id_create_index = "CREATE INDEX IF NOT EXISTS institutionId ON unit (institution_id);"

sqlite_bed_create_query = """
CREATE TABLE IF NOT EXISTS bed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  unit_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  FOREIGN KEY (unit_id) REFERENCES unit (id)
);
"""

sqlite_bed_unit_id_create_index = "CREATE INDEX IF NOT EXISTS unit_id_index ON bed (unit_id);"

sqlite_patient_create_query = """
CREATE TABLE IF NOT EXISTS patient (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mrn INTEGER NULL UNIQUE,
  gender TEXT NULL,
  dob INTEGER NULL,
  first_name TEXT NULL,
  middle_name TEXT NULL,
  last_name TEXT NULL,
  first_seen INTEGER NULL DEFAULT (STRFTIME('%s','NOW')),
  last_updated INTEGER NULL,
  source_id INTEGER DEFAULT 1 NULL,
  weight REAL NULL,
  height REAL NULL,
  FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

sqlite_patient_history_create_query = """
CREATE TABLE IF NOT EXISTS patient_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    value REAL NOT NULL,
    units TEXT NULL,
    time INTEGER NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patient (id) ON DELETE CASCADE
);
"""

sqlite_patient_history_create_index_query = "CREATE INDEX IF NOT EXISTS idx1 ON patient_history (patient_id, field, time DESC);"

sqlite_patient_index_query = """
CREATE INDEX IF NOT EXISTS source_id ON patient (source_id);
"""

sqlite_encounter_create_query = """
CREATE TABLE IF NOT EXISTS encounter (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER,
  bed_id INTEGER,
  start_time INTEGER NOT NULL,
  end_time INTEGER,
  source_id INTEGER DEFAULT 1,
  visit_number TEXT,
  last_updated INTEGER NOT NULL,
  FOREIGN KEY (patient_id) REFERENCES patient (id),
  FOREIGN KEY (bed_id) REFERENCES bed (id),
  FOREIGN KEY (source_id) REFERENCES source (id)
);
"""

sqlite_encounter_create_index_bed_id_query = """
CREATE INDEX IF NOT EXISTS bed_id ON encounter (bed_id);
"""

sqlite_encounter_create_index_patient_id_query = """
CREATE INDEX IF NOT EXISTS patient_id ON encounter (patient_id);
"""

sqlite_encounter_create_index_source_id_query = """
CREATE INDEX IF NOT EXISTS source_id ON encounter (source_id);
"""

sqlite_encounter_insert_trigger = """
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

sqlite_encounter_update_trigger = """
CREATE TRIGGER IF NOT EXISTS encounter_update
    after update
    on encounter
    for each row
BEGIN
UPDATE device_encounter SET end_time=NEW.end_time WHERE encounter_id=NEW.id and end_time IS NULL;
UPDATE device_patient SET end_time=NEW.end_time WHERE patient_id=NEW.patient_id AND start_time=NEW.start_time AND end_time IS NULL;
END;
"""

sqlite_encounter_delete_trigger="""
CREATE TRIGGER IF NOT EXISTS encounter_delete
    after delete
    on encounter
    for each row
BEGIN
DELETE FROM device_patient WHERE patient_id=OLD.patient_id AND start_time=OLD.start_time;
END;
"""

sqlite_device_create_query = """
CREATE TABLE IF NOT EXISTS device (
id INTEGER PRIMARY KEY AUTOINCREMENT,
tag TEXT NOT NULL,
name TEXT NULL,
manufacturer TEXT NULL,
model TEXT NULL,
type TEXT DEFAULT 'static' NOT NULL CHECK (type IN ('static', 'dynamic')),
bed_id INTEGER NULL,
source_id INTEGER DEFAULT 1 NOT NULL,
UNIQUE (tag),
FOREIGN KEY (source_id) REFERENCES source (id),
FOREIGN KEY (bed_id) REFERENCES bed (id)
);
"""

sqlite_device_bed_id_create_index = """
CREATE INDEX IF NOT EXISTS bed_id_index ON device (bed_id);
"""

sqlite_device_source_id_create_index = """
CREATE INDEX IF NOT EXISTS source_id_index ON device (source_id);
"""

sqlite_log_hl7_adt_create_query = """CREATE TABLE IF NOT EXISTS log_hl7_adt(
id INTEGER PRIMARY KEY AUTOINCREMENT,
event_type TEXT NOT NULL,
event_time INTEGER NOT NULL,
mrn TEXT DEFAULT NULL,
first_name TEXT DEFAULT NULL,
middle_name TEXT DEFAULT NULL,
last_name TEXT DEFAULT NULL,
gender TEXT DEFAULT NULL,
dob INTEGER DEFAULT NULL,
visit_num TEXT DEFAULT NULL,
visit_class TEXT DEFAULT NULL,
location TEXT DEFAULT NULL,
previous_location TEXT DEFAULT NULL,
admit_time INTEGER DEFAULT NULL,
discharge_time INTEGER DEFAULT NULL,
source_id INTEGER NOT NULL,
weight REAL NULL,
height REAL NULL,
FOREIGN KEY (source_id) REFERENCES source(id)
);"""

sqlite_log_hl7_adt_source_id_create_index = """CREATE INDEX IF NOT EXISTS source_id
ON log_hl7_adt (source_id);"""

sqlite_device_patient_table = """CREATE TABLE IF NOT EXISTS device_patient (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER NOT NULL,
  patient_id INTEGER NOT NULL,
  start_time INTEGER NOT NULL,
  end_time INTEGER,
  source_id INTEGER DEFAULT 1 NOT NULL,
  FOREIGN KEY (device_id) REFERENCES device(id),
  FOREIGN KEY (patient_id) REFERENCES patient(id),
  FOREIGN KEY (source_id) REFERENCES source(id)
);
"""

sqlite_patient_table_index_1 = "CREATE INDEX IF NOT EXISTS device_patient_index ON device_patient (device_id, patient_id, start_time, end_time);"

sqlite_label_set_create_query = """
CREATE TABLE IF NOT EXISTS label_set (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL UNIQUE,
parent_id INTEGER,
FOREIGN KEY (parent_id) REFERENCES label_set(id)
);
"""

sqlite_label_source_create_query = """
CREATE TABLE IF NOT EXISTS label_source (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT
);
"""

sqlite_label_create_query = """
CREATE TABLE IF NOT EXISTS label (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_set_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    label_source_id INTEGER,
    start_time_n INTEGER NOT NULL,
    end_time_n INTEGER NOT NULL,
    FOREIGN KEY (label_set_id) REFERENCES label_set (id),
    FOREIGN KEY (device_id) REFERENCES device (id),
    FOREIGN KEY (label_source_id) REFERENCES label_source (id)
);
"""

sqlite_label_table_index_1 = "CREATE INDEX IF NOT EXISTS label_idx1 ON label (label_set_id, device_id, start_time_n, end_time_n);"
sqlite_label_table_index_2 = "CREATE INDEX IF NOT EXISTS label_idx1 ON label (label_set_id, device_id, label_source_id, start_time_n, end_time_n);"
