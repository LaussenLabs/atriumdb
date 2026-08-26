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

DEFAULT_UNITS = ''
SUPPORTED_DB_TYPES = ["mysql", "sqlite", "mariadb"]

# Column positions in raw ``measure`` rows returned by the SQL handlers. Both
# handlers select the same explicit column list, so callers can use named
# accessors instead of repeating positional indexes.
MEASURE_ROW_ID = 0
MEASURE_ROW_SIGNAL_KIND = 10
MEASURE_ROW_VALUE_TYPE = 11


def measure_row_signal_kind(row):
    """Return the raw, possibly-null ``signal_kind`` from a measure row."""
    return row[MEASURE_ROW_SIGNAL_KIND]


def measure_row_value_type(row):
    """Return the raw, possibly-null ``value_type`` from a measure row."""
    return row[MEASURE_ROW_VALUE_TYPE]
