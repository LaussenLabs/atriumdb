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
"""No CLI command may echo a credential to stdout.

``atriumdb patient ls`` carried a leftover debug ``print()`` of its whole connection
context -- API token and metadata URI included -- so every invocation wrote a bearer
token, and any password embedded in the URI, into terminal scrollback, shell history
files, CI logs and anything scraping them.

The test drives the command as a user would and asserts the secrets are absent from the
output. The command is expected to fail (there is no dataset at the given location); the
leak happened *before* that failure, so an unsuccessful exit is not a reason to skip the
assertion -- it is the case that matters.

``atriumdb patient show-token`` is excluded on purpose: printing the token is the entire
point of that command, and the test asserts secrets never appear where they were not
asked for.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_cli_does_not_leak_credentials.py -q
"""
import pytest
from click.testing import CliRunner

from atriumdb.cli.atriumdb_cli import cli

SECRET_TOKEN = "tok-must-never-be-printed-9f3a"
SECRET_URI = "mariadb://admin:pw-must-never-be-printed@db.internal:3306/atriumdb"

# (argv, why it takes a connection context)
LISTING_COMMANDS = [
    pytest.param(["patient", "ls"], id="patient-ls"),
    pytest.param(["measure", "ls"], id="measure-ls"),
    pytest.param(["device", "ls"], id="device-ls"),
]


@pytest.mark.parametrize("command", LISTING_COMMANDS)
def test_listing_commands_do_not_print_credentials(command, tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--dataset-location", str(tmp_path / "no-such-dataset"),
        "--metadata-uri", SECRET_URI,
        "--database-type", "sqlite",
        "--endpoint-url", "https://example.invalid/v1",
        "--api-token", SECRET_TOKEN,
        *command,
    ])

    output = result.output + (str(result.exception) if result.exception else "")
    assert SECRET_TOKEN not in output, (
        f"`atriumdb {' '.join(command)}` wrote the API token to its output")
    assert "pw-must-never-be-printed" not in output, (
        f"`atriumdb {' '.join(command)}` wrote the metadata URI password to its output")
