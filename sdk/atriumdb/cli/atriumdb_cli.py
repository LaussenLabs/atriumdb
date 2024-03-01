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
import requests
import qrcodeT
import os
import time
import logging
from tabulate import tabulate
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv, set_key, get_key
from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.adb_functions import parse_metadata_uri
from atriumdb.sql_handler.sql_constants import SUPPORTED_DB_TYPES
from atriumdb.transfer.adb.dataset import transfer_data


_LOGGER = logging.getLogger(__name__)

dotenv_path = "./.env"
load_dotenv(dotenv_path=dotenv_path, override=True)

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
@click.option("--endpoint-url", type=str, required=True, help="The endpoint to connect to for a remote AtriumDB server")
@click.pass_context
def login(ctx, endpoint_url):
    parsed_url = urlparse(endpoint_url)

    # Check if a port is included in the URL
    if parsed_url.port:
        base_url = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}"
    else:
        base_url = f"{parsed_url.scheme}://{parsed_url.hostname}"

    path = parsed_url.path.rstrip('/')

    # Construct the final endpoint URL
    endpoint_url = f"{base_url}{path}"

    # Write endpoint URL to .env file
    load_dotenv(dotenv_path=dotenv_path)
    set_key(dotenv_path, "ATRIUMDB_ENDPOINT_URL", endpoint_url)
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # send get request to the atriumdb api to get the info you need to authenticate with Auth0
    auth_conf_res = requests.get(f'{endpoint_url}/auth/cli/code')

    if auth_conf_res.status_code == 404:
        _LOGGER.error("Invalid Endpoint URL")
        exit()

    if auth_conf_res.status_code != 200:
        _LOGGER.error("Something went wrong")
        exit()

    # information returned from atriumdb API
    auth_conf = auth_conf_res.json()
    auth0_domain = auth_conf['auth0_tenant']
    auth0_client_id = auth_conf['auth0_client_id']
    api_audience = auth_conf['auth0_audience']
    algorithms = auth_conf['algorithms']

    # payload for post request to get the device code from Auth0
    device_code_payload = {
        'client_id': auth0_client_id,
        'scope': 'openid profile offline_access',
        'audience': api_audience
    }

    # send a post request to get the device code
    device_code_response = requests.post(f'https://{auth0_domain}/oauth/device/code', data=device_code_payload)

    if device_code_response.status_code != 200:
        click.echo('Error generating the device code')
        exit(1)

    click.echo('Device code successful')
    device_code_data = device_code_response.json()
    click.echo(f"1. On your computer or mobile device navigate to: \n{device_code_data['verification_uri_complete']}")
    click.echo(f"2. Enter the following code: \n{device_code_data['user_code']}")
    qrcodeT.qrcodeT(device_code_data['verification_uri_complete'])

    # this is the payload to get a token for "input constrained devices"
    token_payload = {
        'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
        'device_code': device_code_data['device_code'],
        'client_id': auth0_client_id
    }
    authenticated = False

    # Calculate absolute expiration time
    expiration_time = time.time() + int(device_code_data['expires_in'])

    while not authenticated and time.time() < expiration_time:
        token_response = requests.post(f'https://{auth0_domain}/oauth/token', data=token_payload)
        token_data = token_response.json()

        if token_response.status_code == 200:
            click.echo('Authenticated!')
            authenticated = True

            # set the new keys in the .env file
            set_key(dotenv_path, "ATRIUMDB_API_TOKEN", token_data['access_token'])
            set_key(dotenv_path, "ATRIUMDB_API_REFRESH_TOKEN", token_data['refresh_token'])
            set_key(dotenv_path, "ATRIUMDB_DATABASE_TYPE", "api")

            click.echo("Your API Token is:")
            click.echo(f"{token_data['access_token']} \n")
            click.echo("Your Refresh Token is:")
            click.echo(f"{token_data['refresh_token']} \n")
            click.echo("The variables ATRIUMDB_API_TOKEN and ATRIUMDB_API_REFRESH_TOKEN have been set in your .env file")

            # load environment variables into the OS
            load_dotenv(dotenv_path=dotenv_path, override=True)

        elif token_data['error'] not in ('authorization_pending', 'slow_down'):
            click.echo(token_data['error_description'])
            exit(1)
        else:
            time.sleep(device_code_data['interval'])

    # Check if the authentication process timed out
    if not authenticated:
        click.echo("Authentication Request Timed-Out")


@click.command(help="Refresh the API token using the stored endpoint URL.")
@click.pass_context
def refresh_token(ctx):
    # Load .env file
    load_dotenv(dotenv_path=dotenv_path)

    # Check if endpoint URL is set
    endpoint_url = get_key(dotenv_path, "ATRIUMDB_ENDPOINT_URL")

    if not endpoint_url:
        click.echo("Endpoint URL not set. Please use 'atriumdb login --endpoint-url MY_URL'.")
        exit(1)

    # Call the login function with the stored endpoint URL
    ctx.invoke(login, endpoint_url=endpoint_url)


@click.command(help="Displays the current CLI configuration.")
@click.pass_context
def config(ctx):
    # Retrieve the configuration from the context
    config_data = {
        "Endpoint URL": "Not Set" if not ctx.obj.get("endpoint_url") else ctx.obj.get("endpoint_url"),
        "API Token": "Not Set" if not ctx.obj.get("api_token") else ctx.obj.get("api_token"),
        "Dataset Location": "Not Set" if not ctx.obj.get("dataset_location") else ctx.obj.get("dataset_location"),
        "Metadata URI": "Not Set" if not ctx.obj.get("metadata_uri") else ctx.obj.get("metadata_uri"),
        "Database Type": "Not Set" if not ctx.obj.get("database_type") else ctx.obj.get("database_type"),
    }

    if config_data["API Token"] != "Not Set":
        config_data["API Token"] = config_data["API Token"][:15] + "..."

    # Determine the mode (remote or local)
    mode = "Remote" if config_data["Database Type"] == "api" else "Local"

    # Prepare data for tabulation
    data = [
        ["Mode", mode],
        *config_data.items()
    ]

    # Print the tabulated data
    click.echo(tabulate(data, headers=["Configuration", "Value"], tablefmt="grid"))
    if config_data["API Token"] != "Not Set":
        click.echo("Full API Token:")
        click.echo(ctx.obj.get("api_token", "Not Set"))


@click.command()
@click.pass_context
@click.argument('definition_filename', type=click.Path(exists=True))
@click.option("--gap-tolerance", type=int, default=None,
              help="The time (in --time-units units) defining the minimum gap between different sections of data. Large numbers will "
                   "increase efficiency by performing large queries, but may take in extra unwanted data. Leaving the "
                   "option as the default 5 minutes strikes a good balance.")
@click.option("--deidentify", type=click.Choice(['True', 'False', 'filename']), default='False',
              help="Whether to deidentify patient data. Can also specify a filename for custom ID mapping.")
@click.option("--patient-cols", type=str, multiple=True, default=None,
              help="List patient columns to transfer, valid options are: "
                   "mrn, gender, dob, first_name, middle_name, last_name, first_seen, "
                   "last_updated, source_id, weight, height")
@click.option("--block-size", type=int, default=None, help="The target number of values per compression block.")
@click.option("--include-labels", is_flag=True, default=True,
              help="Whether to include labels in the data transfer.")
@click.option("--measure-tag-match-rule", type=click.Choice(['all', 'best']), default=None,
              help="Determines how to match the measures by tags.")
@click.option("--time-shift", type=int, default=None,
              help="Amount of time (in the specified units) by which to shift all timestamps in the data.")
@click.option("--time-units", type=click.Choice(['ns', 'us', 'ms', 's']), default='s',
              help="Units for time-related parameters like gap-tolerance and time-shift. Defaults to seconds ('s').")
@click.option("--export-format", type=click.Choice(['tsc', 'csv', 'npz', 'wfdb']), default='tsc',
              help="The format used for exporting data. Default is 'tsc'.")
@click.option("--dataset-location-out", type=click.Path(), help="Path to export directory",
              envvar='ATRIUMDB_EXPORT_DATASET_LOCATION')
@click.option("--metadata-uri-out", type=str, help="When exporting in tsc format with a mariadb supported dataset, The URI of a metadata server",
              envvar='ATRIUMDB_METADATA_URI_OUT')
@click.option("--database-type-out", type=str, help="The metadata database type",
              envvar='ATRIUMDB_DATABASE_TYPE_OUT')
def export(ctx, definition_filename, gap_tolerance, deidentify, patient_cols, block_size, include_labels, measure_tag_match_rule,
           time_shift, time_units, export_format, dataset_location_out, metadata_uri_out, database_type_out):

    patient_info_to_transfer = list(patient_cols) if patient_cols else None
    deidentify = deidentify if deidentify.lower() in ['true', 'false'] else Path(deidentify)

    endpoint_url = ctx.obj["endpoint_url"]
    api_token = ctx.obj["api_token"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]
    database_type = ctx.obj["database_type"]

    src_sdk = get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, endpoint_url, api_token)
    dest_sdk = get_sdk_from_cli_params(dataset_location_out, metadata_uri_out, database_type_out, None, None)

    if block_size:
        dest_sdk.block.block_size = block_size

    definition = DatasetDefinition(filename=definition_filename)

    transfer_data(src_sdk, dest_sdk, definition, export_format=export_format, gap_tolerance=gap_tolerance,
                  deidentify=deidentify, patient_info_to_transfer=patient_info_to_transfer, include_labels=include_labels,
                  measure_tag_match_rule=measure_tag_match_rule, time_shift=time_shift, time_units=time_units)


def get_sdk_from_cli_params(dataset_location, metadata_uri, database_type, api_url, api_token):
    connection_params = None if metadata_uri is None else parse_metadata_uri(metadata_uri)
    if database_type != "api" and not (Path(dataset_location) / "tsc").is_dir():
        return AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=database_type,
                                 connection_params=connection_params)

    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=database_type,
                    connection_params=connection_params, api_url=api_url, token=api_token)
    return sdk


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
cli.add_command(atriumdb_patient)
cli.add_command(atriumdb_measure)
cli.add_command(atriumdb_device)
cli.add_command(login)
cli.add_command(refresh_token)
cli.add_command(config)


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
