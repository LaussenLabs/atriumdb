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

import click
from tabulate import tabulate
from pathlib import Path

import requests
import qrcodeT
import urllib3
import os
import time

from auth0.authentication.token_verifier import TokenVerifier, AsymmetricSignatureVerifier

from dotenv import load_dotenv

from atriumdb import AtriumSDK
from atriumdb.adb_functions import parse_metadata_uri
from atriumdb.cli.sdk import get_sdk_from_env_vars, create_sdk_from_env_vars
from atriumdb.sql_handler.sql_constants import SUPPORTED_DB_TYPES
from atriumdb.transfer.cohort.cohort_yaml import parse_atrium_cohort_yaml
from atriumdb.transfer.adb.dataset import transfer_data
from atriumdb.transfer.formats.dataset import export_dataset, import_dataset
from atriumdb.transfer.formats.export_data import export_data_from_sdk
from atriumdb.transfer.formats.import_data import import_data_to_sdk

import logging

_LOGGER = logging.getLogger(__name__)

load_dotenv(dotenv_path="./.env", override=True)

cli_help_text = """
The atriumdb command is a command line interface for the Atrium database, 
allowing you to import and export data from the database.

The import subcommand is used to import data into the database from common formats, while the export subcommand 
is used to export data from the database to common formats.

The SDK data is defined by the following environment variables or corresponding command line options:

ATRIUMDB_DATASET_LOCATION (--dataset-location): This environment variable specifies the directory location where the Atrium SDK data will 
be stored.

ATRIUMDB_METADATA_URI (--metadata-uri): This environment variable specifies the URI of a metadata server, which includes the db_type, host, password, db_name, and other relevant connection details. The format should be "<db_type>://<user>:<password>@<host>:<port>/<database_name>".

ATRIUMDB_DATABASE_TYPE (--database-type): This environment variable specifies the type of metadata database supporting the dataset. The possible values are "sqlite", "mariadb", "mysql" or "api".

ATRIUMDB_ENDPOINT_URL (--endpoint-url): This environment variable specifies the endpoint to connect to for a remote AtriumDB server.

ATRIUMDB_API_TOKEN (--api-token): This environment variable specifies a token to authorize api access.

You can use command line options in place of the environment variables for a more flexible configuration.

For more information on how to use these subcommands 
and the Atrium database, please visit the documentation at https://atriumdb.sickkids.ca/docs/.
"""


@click.group(help=cli_help_text)
@click.option("--dataset-location", type=click.Path(), envvar="ATRIUMDB_DATASET_LOCATION",
              default=None, help="The local path to a dataset")
@click.option("--metadata-uri", type=str, envvar="ATRIUMDB_METADATA_URI",
              default=None, help="The URI of a metadata server")
@click.option("--database-type", type=click.Choice(SUPPORTED_DB_TYPES + ["api"]), envvar="ATRIUMDB_DATABASE_TYPE",
              default=None, help="The type of metadata database supporting the dataset.")
@click.option("--endpoint-url", type=str, envvar="ATRIUMDB_ENDPOINT_URL",
              default=None, help="The endpoint to connect to for a remote AtriumDB server")
@click.option("--api-token", type=str, envvar="ATRIUMDB_API_TOKEN",
              default=None, help="A token to authorize api access.")
@click.pass_context
def cli(ctx, dataset_location, metadata_uri, database_type, endpoint_url, api_token):
    ctx.ensure_object(dict)

    ctx.obj["endpoint_url"] = endpoint_url
    ctx.obj["api_token"] = api_token
    ctx.obj["dataset_location"] = dataset_location
    ctx.obj["metadata_uri"] = metadata_uri
    ctx.obj["database_type"] = database_type


@click.command(help="Endpoint to login with QR code.")
@click.pass_context
def login(ctx):
    endpoint_url = ctx.obj["endpoint_url"]

    endpoint_url = endpoint_url.rstrip('/')

    auth_conf_res = requests.get(f'{endpoint_url}/auth/cli/code')

    if auth_conf_res.status_code == 404:
        _LOGGER.error("Invalid Endpoint URL")
        exit()

    if auth_conf_res.status_code != 200:
        _LOGGER.error("Something went wrong")
        exit()

    auth_conf = auth_conf_res.json()
    auth0_domain = auth_conf['auth0_tenant']
    auth0_client_id = auth_conf['auth0_client_id']
    api_audience = auth_conf['auth0_audience']
    algorithms = auth_conf['algorithms']

    def validate_token(id_token):
        jwks_url = f'https://{auth0_domain}/.well-known/jwks.json'
        issuer = f'https://{auth0_domain}/'
        sv = AsymmetricSignatureVerifier(jwks_url)
        tv = TokenVerifier(signature_verifier=sv, issuer=issuer, audience=auth0_client_id)
        tv.verify(id_token)

    device_code_payload = {
        'client_id': auth0_client_id,
        'scope': 'openid profile',
        'audience': api_audience
    }
    device_code_response = requests.post(f'https://{auth0_domain}/oauth/device/code', data=device_code_payload)

    if device_code_response.status_code != 200:
        click.echo('Error generating the device code')
        exit(1)

    click.echo('Device code successful')
    device_code_data = device_code_response.json()
    click.echo(f"1. On your computer or mobile device navigate to: \n{device_code_data['verification_uri_complete']}")
    click.echo(f"2. Enter the following code: \n{device_code_data['user_code']}")
    qrcodeT.qrcodeT(device_code_data['verification_uri_complete'])

    token_payload = {
        'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
        'device_code': device_code_data['device_code'],
        'client_id': auth0_client_id
    }

    authenticated = False
    while not authenticated:
        # click.echo('Checking if the user completed the flow...')
        token_response = requests.post(f'https://{auth0_domain}/oauth/token', data=token_payload)

        token_data = token_response.json()
        if token_response.status_code == 200:
            click.echo('Authenticated!')

            validate_token(token_data['id_token'])

            authenticated = True

            set_env_var_in_dotenv("ATRIUMDB_API_TOKEN", token_data['access_token'])
            set_env_var_in_dotenv("ATRIUMDB_DATABASE_TYPE", "api")
            click.echo("Your API Token is:\n")
            click.echo(token_data['access_token'])
            click.echo("The variable ATRIUMDB_API_TOKEN has been set in your .env file and")
            load_dotenv(dotenv_path="./.env", override=True)

        elif token_data['error'] not in ('authorization_pending', 'slow_down'):
            click.echo(token_data['error_description'])
            exit(1)
        else:
            time.sleep(device_code_data['interval'])


@click.command()
@click.pass_context
@click.option("--format", "export_format", default="adb", help="Format of the exported data",
              type=click.Choice(["adb", "csv", "parquet", "numpy", "wfdb"]))
@click.option("--packaging-type", default="files", help="Type of packaging for the exported data",
              type=click.Choice(["files", "tar", "gzip"]))
@click.option("--cohort-file", type=click.Path(), help="Cohort file for automatically configuring export parameters")
@click.option("--measure-ids", type=int, multiple=True, default=None, help="List of measure ids to export")
@click.option("--measures", type=str, multiple=True, default=None, help="List of measure tags to export")
@click.option("--device-ids", type=int, multiple=True, default=None, help="List of device ids to export")
@click.option("--devices", type=str, multiple=True, default=None, help="List of device tags to export")
@click.option("--patient-ids", type=int, multiple=True, default=None, help="List of patient ids to export")
@click.option("--mrns", type=str, multiple=True, default=None, help="List of MRNs to export")
@click.option("--start-time", type=int, help="Start time for exporting data")
@click.option("--end-time", type=int, help="End time for exporting data")
@click.option("--dataset-location-out", type=click.Path(), help="Path to export directory",
              envvar='ATRIUMDB_EXPORT_DATASET_LOCATION')
@click.option("--metadata-uri-out", type=str, help="The URI of a metadata server",
              envvar='ATRIUMDB_METADATA_URI_OUT')
@click.option("--database-type-out", type=str, help="The metadata database type",
              envvar='ATRIUMDB_DATABASE_TYPE_OUT')
@click.option("--by-patient", type=bool, default=False, help="Whether or not to include patient mapping")
def export(ctx, export_format, packaging_type, cohort_file, measure_ids, measures, device_ids, devices, patient_ids,
           mrns, start_time, end_time, dataset_location_out, metadata_uri_out, database_type_out, by_patient):
    measure_ids = None if measure_ids is not None and len(measure_ids) == 0 else list(measure_ids)
    measures = None if measures is not None and len(measures) == 0 else list(measures)
    device_ids = None if device_ids is not None and len(device_ids) == 0 else list(device_ids)
    devices = None if devices is not None and len(devices) == 0 else list(devices)
    patient_ids = None if patient_ids is not None and len(patient_ids) == 0 else list(patient_ids)
    mrns = None if mrns is not None and len(mrns) == 0 else list(mrns)

    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    # If format is not adb, do something else
    implemented_formats = ["adb", "csv"]
    if export_format not in implemented_formats:
        raise NotImplementedError(f"Only {implemented_formats} formats are currently supported for export")

    from_sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, endpoint_url, api_token)

    # If a cohort file is specified, parse it and use its parameters for transfer_data
    if cohort_file:
        if cohort_file.endswith((".yml", ".yaml")):
            cohort_params = parse_atrium_cohort_yaml(cohort_file)
            measures = cohort_params["measures"]
            measure_ids = cohort_params["measure_ids"]
            devices = cohort_params["devices"]
            device_ids = cohort_params["device_ids"]
            patient_ids = cohort_params["patient_ids"]
            mrns = cohort_params["mrns"]
            start_time = cohort_params["start_epoch_s"]
            end_time = cohort_params["end_epoch_s"]

        else:
            raise ValueError("Unsupported cohort file format")

    if measures is not None:
        measure_ids = [] if measure_ids is None else measure_ids

        match_measure_ids = []
        for measure_tag in measures:
            match_dict = from_sdk.search_measures(tag_match=measure_tag)
            match_measure_ids.extend(list(match_dict.keys()))

        measure_ids.extend(match_measure_ids)
        measure_ids = list(set(measure_ids))

    if devices is not None:
        device_ids = [] if device_ids is None else device_ids

        match_device_ids = []
        for device_tag in devices:
            match_dict = from_sdk.search_devices(tag_match=device_tag)
            match_device_ids.extend(list(match_dict.keys()))

        device_ids.extend(match_device_ids)
        device_ids = list(set(device_ids))

    if export_format == "adb":
        assert dataset_location_out is not None, "dataset-location-out option or ATRIUMDB_EXPORT_DATASET_LOCATION " \
                                                 "envvar must be specified."

        # Transfer the data using the specified parameters
        to_sdk = get_sdk_from_cli_params(dataset_location_out, metadata_uri_out, database_type_out, None,
                                         None)

        transfer_data(from_sdk=from_sdk, to_sdk=to_sdk, measure_id_list=measure_ids, device_id_list=device_ids,
                      patient_id_list=patient_ids, mrn_list=mrns, start=start_time, end=end_time)

    else:
        assert dataset_location_out is not None, "dataset-location-out option or ATRIUMDB_EXPORT_DATASET_LOCATION " \
                                                 "envvar must be specified."

        dataset_dir = Path(dataset_location_out) / f"{export_format}_export"
        export_dataset(sdk=from_sdk, directory=dataset_dir, data_format=export_format, device_id_list=device_ids,
                       patient_id_list=patient_ids, mrn_list=mrns, start=start_time, end=end_time,
                       by_patient=by_patient, measure_id_list=measure_ids)


def get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, api_url, api_token):
    connection_params = None if metadata_uri is None else parse_metadata_uri(metadata_uri)
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=database_type,
                    connection_params=connection_params, api_url=api_url, token=api_token)
    return sdk


@click.command(name="import")
@click.pass_context
@click.option("--format", "import_format", default="adb", help="Format of the imported data",
              type=click.Choice(["adb", "csv", "parquet", "numpy", "wfdb"]))
@click.option("--packaging-type", default="files", help="Type of packaging for the imported data",
              type=click.Choice(["files", "tar", "gzip"]))
@click.option("--dataset-location-in", type=click.Path(), help="Path to import directory",
              envvar='ATRIUMDB_IMPORT_DATASET_LOCATION')
@click.option("--metadata-uri-in", type=str, help="The URI of a metadata server to import from",
              envvar='ATRIUMDB_IMPORT_METADATA_URI', default=None)
@click.option("--endpoint-url-in", type=str,
              help="The endpoint to connect to for a remote AtriumDB server to import from",
              envvar='ATRIUMDB_IMPORT_ENDPOINT_URL', default=None)
@click.option("--measure-ids", type=int, multiple=True, default=None, help="List of measure ids to import")
@click.option("--measures", type=str, multiple=True, default=None, help="List of measure tags to import")
@click.option("--device-ids", type=int, multiple=True, default=None, help="List of device ids to import")
@click.option("--devices", type=str, multiple=True, default=None, help="List of device tags to import")
@click.option("--patient-ids", type=int, multiple=True, default=None, help="List of patient ids to import")
@click.option("--mrns", type=str, multiple=True, default=None, help="List of MRNs to import")
@click.option("--start-time", type=int, help="Start time for importing data")
@click.option("--end-time", type=int, help="End time for importing data")
def import_(ctx, import_format, packaging_type, dataset_location_in, metadata_uri_in, endpoint_url_in, measure_ids,
            measures, device_ids, devices, patient_ids, mrns, start_time, end_time):
    measure_ids = None if measure_ids is not None and len(measure_ids) == 0 else list(measure_ids)
    measures = None if measures is not None and len(measures) == 0 else list(measures)
    device_ids = None if device_ids is not None and len(device_ids) == 0 else list(device_ids)
    devices = None if devices is not None and len(devices) == 0 else list(devices)
    patient_ids = None if patient_ids is not None and len(patient_ids) == 0 else list(patient_ids)
    mrns = None if mrns is not None and len(mrns) == 0 else list(mrns)

    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    implemented_formats = ["csv", "parquet"]

    if import_format not in implemented_formats:
        raise NotImplementedError(f"Only {implemented_formats} formats are currently supported for import")

    if dataset_location_in is None:
        raise ValueError("dataset-location-in option or ATRIUMDB_IMPORT_DATASET_LOCATION envvar must be specified.")

    if not Path(dataset_location_in).is_dir():
        raise ValueError(f"{dataset_location_in} is not a valid directory.")

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, None, None)

    import_dataset(sdk, directory=dataset_location_in, data_format=import_format)


@click.group(name='measure')
@click.pass_context
def atriumdb_measure(ctx):
    pass


@atriumdb_measure.command(name="ls")
@click.pass_context
@click.option("--tag-match", type=str, help="Filter measures by tag string match")
@click.option("--name-match", type=str, help="Filter measures by name string match")
@click.option("--unit", type=str, help="Filter measures by units")
@click.option("--freq", type=float, help="Filter measures by frequency")
@click.option("--freq-units", type=str, help="Unit of frequency", default="Hz")
@click.option("--source-id", type=int, help="Filter measures by source identifier")
def measure_ls(ctx, tag_match, name_match, unit, freq, freq_units, source_id):
    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, endpoint_url, api_token)
    result = sdk.search_measures(
        tag_match=tag_match, freq=freq, unit=unit, name_match=name_match, freq_units=freq_units)

    headers = ["Measure ID", "Tag", "Name", "Frequency (nHz)", "Code", "Unit", "Unit Label", "Unit Code", "Source ID"]
    table = []

    for measure_id, measure_info in result.items():
        row = [measure_id]
        for key, value in measure_info.items():
            row.append(value)
        table.append(row)

    click.echo("\nMeasures:")
    click.echo(tabulate(table, headers=headers))


@click.group(name="device")
@click.pass_context
def atriumdb_device(ctx):
    pass


@atriumdb_device.command(name="ls")
@click.pass_context
@click.option("--tag-match", type=str, help="Filter devices by tag string match")
@click.option("--name-match", type=str, help="Filter devices by name string match")
@click.option("--manufacturer-match", type=str, help="Filter devices by manufacturer string match")
@click.option("--model-match", type=str, help="Filter devices by model string match")
@click.option("--bed-id", type=str, help="Filter devices by bed identifier")
@click.option("--source-id", type=str, help="Filter devices by source identifier")
def device_ls(ctx, tag_match, name_match, manufacturer_match, model_match, bed_id, source_id):
    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, endpoint_url, api_token)
    result = sdk.search_devices(tag_match=tag_match, name_match=name_match)

    headers = ["Device ID", "Tag", "Name", "Manufacturer", "Model", "Type", "Bed ID", "Source ID"]
    table = []

    for device_id, device_info in result.items():
        row = [device_id]
        for key, value in device_info.items():
            row.append(value)
        table.append(row)

    click.echo("\nDevices:")
    click.echo(tabulate(table, headers=headers))


@click.group(name="patient")
@click.pass_context
def atriumdb_patient(ctx):
    pass


@atriumdb_patient.command(name="ls")
@click.pass_context
@click.option("--skip", type=int, help="Offset number of patients to return")
@click.option("--limit", type=int, help="Limit number of patients to return")
@click.option("--age-years-min", type=float, help="Filter patients by minimum age in years")
@click.option("--age-years-max", type=float, help="Filter patients by maximum age in years")
@click.option("--gender", type=str, help="Filter patients by gender")
@click.option("--source-id", type=str, help="Filter patients by source identifier")
@click.option("--first-seen", type=int, help="Filter patients by first seen timestamp in epoch time")
@click.option("--last-updated", type=int, help="Filter patients by last updated timestamp in epoch time")
def patient_ls(ctx, skip, limit, age_years_min, age_years_max, gender, source_id, first_seen, last_updated):
    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    print(dataset_location, metadata_uri, database_type, endpoint_url, api_token)

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, endpoint_url, api_token)

    result = sdk.get_all_patients(skip=skip, limit=limit)

    if len(result) == 0:
        return

    headers = list(list(result.values())[0].keys())
    table = []

    for patient_id, patient_info in result.items():
        table.append(list(patient_info.values()))

    click.echo("\nPatients:")
    click.echo(tabulate(table, headers=headers))


cli.add_command(export)
cli.add_command(import_)
cli.add_command(atriumdb_patient)
cli.add_command(atriumdb_measure)
cli.add_command(atriumdb_device)
cli.add_command(login)


def set_env_var_in_dotenv(name, value):
    dotenv_file = ".env"
    env_vars = {}

    if os.path.exists(dotenv_file):
        with open(dotenv_file, "r") as file:
            for line in file:
                if line.strip() and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    env_vars[key] = val

    env_vars[name] = value

    with open(dotenv_file, "w") as file:
        for key, val in env_vars.items():
            file.write(f"{key}={val}\n")
