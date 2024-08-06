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
import csv

import numpy as np

from atriumdb.transfer.adb.datestring_conversion import nanoseconds_to_date_string_with_tz

time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}


def _write_csv(file_path, times: np.ndarray, values: np.ndarray, measure_tag, export_time_format='ns', timezone_str=None):
    # Determine the header based on the export_time_format
    if export_time_format == 'date':
        header_time = "Timestamp (Date)"
    else:
        header_time = f"Timestamp ({export_time_format})"

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([header_time, measure_tag])

        # Convert times all at once if not exporting as dates
        if export_time_format in time_unit_options:
            conversion_factor = time_unit_options[export_time_format]
            converted_times = times / conversion_factor
            for time, value in zip(converted_times, values):
                writer.writerow([time, value])
        elif export_time_format == 'date':
            # For dates, convert each time individually
            for time, value in zip(times, values):
                date_string = nanoseconds_to_date_string_with_tz(time, timezone_str=timezone_str)
                writer.writerow([date_string, value])
