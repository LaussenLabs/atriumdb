from atriumdb import AtriumSDK
from click.testing import CliRunner
import shutil

from atriumdb.adb_functions import generate_metadata_uri
from atriumdb.cli.atriumdb_cli import cli
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-transfer'
MAX_RECORDS = 1


def test_transfer_cli():
    _test_for_both(DB_NAME, _test_transfer_cli)


def _test_transfer_cli(db_type, dataset_location, connection_params):
    runner = CliRunner()
    dataset_location_out = str(dataset_location) + "_2"
    shutil.rmtree(dataset_location_out, ignore_errors=True)

    connection_params_out = None if connection_params is None else connection_params.copy()
    if db_type in ['mysql', 'mariadb']:
        connection_params_out['database'] += "-2"
        host = connection_params_out['host']
        user = connection_params_out['user']
        password = connection_params_out['password']
        db_name = connection_params_out['database']
        port = connection_params_out['port']

        maria_handler = MariaDBHandler(host, user, password, db_name)

        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")

    metadata_uri = None if connection_params is None else generate_metadata_uri(connection_params)
    metadata_uri_out = None if connection_params_out is None else generate_metadata_uri(connection_params_out)

    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = AtriumSDK.create_dataset(
        dataset_location=dataset_location_out, database_type=db_type, connection_params=connection_params_out)

    write_mit_bih_to_dataset(sdk_1, max_records=MAX_RECORDS)

    result = runner.invoke(cli, [
        "--dataset-location", dataset_location,
        "--metadata-uri", metadata_uri,
        "--database-type", db_type,
        "export",
        "--format", "adb",
        "--dataset-location-out", dataset_location_out,
        "--metadata-uri-out", metadata_uri_out
    ])

    assert result.exit_code == 0, str(result.exc_info)

    assert_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS)
