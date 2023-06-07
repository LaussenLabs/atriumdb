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

from sqlalchemy import MetaData, Table, Column, Integer, BigInteger, String, ForeignKey, UniqueConstraint


MEASURE_TABLE_NAME = "measures"
DEVICE_TABLE_NAME = "devices"
SETTINGS_TABLE_NAME = "settings"
BLOCK_TABLE_NAME = "block_index"
FILE_TABLE_NAME = "file_index"
INTERVAL_TABLE_NAME = "interval_index"
ENCOUNTER_TABLE_NAME = "encounter"


def generate_metadata():
    metadata = MetaData()

    Table(MEASURE_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('measure_tag', String(64), nullable=True),
          Column('measure_name', String(190), nullable=True),
          Column('freq_nhz', BigInteger),
          Column('units', String(64), nullable=True),
          UniqueConstraint('measure_tag', 'freq_nhz', 'units'))

    Table(DEVICE_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('device_tag', String(64), nullable=True, unique=True),
          Column('device_name', String(190), nullable=True))

    Table(FILE_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('measure_id', Integer, ForeignKey("measures.id")),
          Column('device_id', Integer, ForeignKey("devices.id")),
          Column('path', String(190), unique=True))

    Table(BLOCK_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('measure_id', Integer, ForeignKey("measures.id")),
          Column('device_id', Integer, ForeignKey("devices.id")),
          Column('file_id', Integer, ForeignKey("file_index.id")),
          Column('start_byte', BigInteger),
          Column('num_bytes', BigInteger),
          Column('start_time_n', BigInteger),
          Column('end_time_n', BigInteger),
          Column('num_values', BigInteger))

    Table(INTERVAL_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('measure_id', Integer, ForeignKey("measures.id")),
          Column('device_id', Integer, ForeignKey("devices.id")),
          Column('start_time_n', BigInteger),
          Column('end_time_n', BigInteger))

    Table("session",
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('hash', String(32), unique=True))

    Table("ingest",
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('session_id', Integer, ForeignKey("session.id")),
          Column('hash', String(32), unique=True))

    Table(ENCOUNTER_TABLE_NAME,
          metadata,
          Column('id', Integer, primary_key=True, autoincrement=True),
          Column('patient_id', Integer),
          Column('device_id', Integer, ForeignKey("devices.id")),
          Column('start_time_n', BigInteger),
          Column('end_time_n', BigInteger))

    Table(SETTINGS_TABLE_NAME,
          metadata,
          Column('setting_name', String(64), primary_key=True),
          Column('setting_value', String(64)))

    return metadata
