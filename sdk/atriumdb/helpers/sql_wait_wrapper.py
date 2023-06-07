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

from sqlalchemy.exc import OperationalError
import time
import functools

DB_LOCKED_STRING = "database is locked"


def sql_lock_wait(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        while True:
            try:
                return fn(*args, **kwargs)
            except OperationalError as e:
                error_string = str(e)
                if DB_LOCKED_STRING in error_string:
                    time.sleep(0.1)
                    continue
                else:
                    raise e
    return wrap
