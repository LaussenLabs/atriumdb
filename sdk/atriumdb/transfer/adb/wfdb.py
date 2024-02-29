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
import numpy as np


def _ingest_data_wfdb(headers, times, values, file_path, measure_tag, freq_hz, measure_units):
    try:
        import wfdb
    except ImportError as e:
        raise ImportError("the package wfdb is required for exporting in WFDB format. "
                          "Please install wfdb to proceed.") from e
    # Convert values to the required format
    values = values.astype(np.float64).reshape(-1, 1)

    # Get the filename (including its extension) and the parent directory
    file_name = file_path.stem
    write_dir = file_path.parent

    wfdb.wrsamp(file_name, fs=freq_hz, units=[measure_units], sig_name=[measure_tag], p_signal=values,
                write_dir=str(write_dir))
