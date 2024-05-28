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
import datetime


def nanoseconds_to_date_string_with_tz(nanoseconds):
    # Convert nanoseconds to seconds
    seconds = nanoseconds / 1e9
    # Create a timezone-aware datetime object from the epoch
    dt_utc = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    # Get the local system's timezone
    local_tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    # Convert the datetime to the local timezone
    dt_local = dt_utc.astimezone(local_tz)
    # Format the datetime as a string with timezone information and microseconds
    date_string_with_tz = dt_local.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    return date_string_with_tz
