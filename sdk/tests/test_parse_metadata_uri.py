import pytest

from atriumdb.adb_functions import parse_metadata_uri


def test_parse_metadata_uri():
    # valid metadata_uris
    metadata_uri = "mysql://username:password@host:port/dbname"
    components = parse_metadata_uri(metadata_uri)
    assert components == {
        'sqltype': 'mysql',
        'user': 'username',
        'password': 'password',
        'host': 'host',
        'port': 'port',
        'database': 'dbname'
    }

    metadata_uri = "postgresql://username:password@host:port"
    components = parse_metadata_uri(metadata_uri)
    assert components == {
        'sqltype': 'postgresql',
        'user': 'username',
        'password': 'password',
        'host': 'host',
        'port': 'port',
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
