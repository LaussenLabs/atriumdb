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

import pytest

from atriumdb.adb_functions import parse_metadata_uri, generate_metadata_uri


def test_parse_metadata_uri():
    # valid metadata_uris
    metadata_uri = "mysql://username:password@host:3306/dbname"
    components = parse_metadata_uri(metadata_uri)
    assert components == {
        'sqltype': 'mysql',
        'user': 'username',
        'password': 'password',
        'host': 'host',
        'port': 3306,
        'database': 'dbname'
    }

    metadata_uri = "postgresql://username:password@host:3306"
    components = parse_metadata_uri(metadata_uri)
    assert components == {
        'sqltype': 'postgresql',
        'user': 'username',
        'password': 'password',
        'host': 'host',
        'port': 3306,
        'database': None
    }

    # invalid metadata_uris
    metadata_uri = "username:password@host:port"
    with pytest.raises(ValueError):
        parse_metadata_uri(metadata_uri)

    metadata_uri = "mysql://username:password@host"
    with pytest.raises(ValueError):
        parse_metadata_uri(metadata_uri)

    metadata_uri = "postgresql://username@host:port/dbname?params=value"
    with pytest.raises(ValueError):
        parse_metadata_uri(metadata_uri)

    # Test the reverse
        # Test case 1: with a database name
        metadata = {
            'sqltype': 'mysql',
            'user': 'username',
            'password': 'password',
            'host': 'host',
            'port': 3306,
            'database': 'dbname'
        }
        expected_uri = "mysql://username:password@host:3306/dbname"
        generated_uri = generate_metadata_uri(metadata)
        assert generated_uri == expected_uri

        # Test case 2: without a database name
        metadata = {
            'sqltype': 'mysql',
            'user': 'username',
            'password': 'password',
            'host': 'host',
            'port': 3306,
        }
        expected_uri = "mysql://username:password@host:3306"
        generated_uri = generate_metadata_uri(metadata)
        assert generated_uri == expected_uri
