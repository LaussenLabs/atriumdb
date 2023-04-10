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
