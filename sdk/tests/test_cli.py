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

    pass


def test_import_export_cli():
    connection_params, dataset_location, metadata_connection_type = get_sdk_params_from_env_vars()
    if metadata_connection_type in ["mariadb", "mysql"]:
        host = connection_params['host']
        user = connection_params['user']
        password = connection_params['password']
        db_name = connection_params['database']
        maria_handler = MariaDBHandler(host, user, password, db_name)
        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        shutil.rmtree(dataset_location, ignore_errors=True)
    runner = CliRunner()

    # Import data from a CSV file
    filename = Path(__file__).parent / "example_data" / "import_data_1.formats"
    imported_df = pd.read_csv(filename)

    result = runner.invoke(import_, ["--input-file", str(filename)])
    assert result.exit_code == 0, f"Import failed with error: {result.output}"
    print()
    print(result.output)

    # Export data to a CSV file
    export_filename = Path(__file__).parent / "example_data" / "export_data_1.formats"
    result = runner.invoke(export, ["--output-file", str(export_filename),
                                        "--measure-id", "1",
                                        "--device-id", "1",
                                        "--start-time", "0",
                                        "--end-time", str(10 ** 62)])
    assert result.exit_code == 0, f"Export failed with error: {result.output}"
    print()
    print(result.output)

    # Check if the imported and exported data are the same
    exported_df = pd.read_csv(export_filename)

    assert imported_df.equals(exported_df), "Imported and exported data are not the same"
