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

from click.testing import CliRunner

from atriumdb.cli.atriumdb_cli import export, import_
from atriumdb.cli.hello import hello

from pathlib import Path
import pandas as pd
import shutil

from atriumdb.cli.sdk import get_sdk_params_from_env_vars
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler


def test_hello_cli():
    runner = CliRunner()
    result = runner.invoke(hello)

    print()
    print(result.output)
    assert result.output == "Hello, World!\n"
