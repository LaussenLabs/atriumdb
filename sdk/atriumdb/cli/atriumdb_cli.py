import click
from tabulate import tabulate
from pathlib import Path

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

load_dotenv()

_LOGGER = logging.getLogger(__name__)

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
              help="The local path to a dataset")
@click.option("--metadata-uri", type=str, envvar="ATRIUMDB_METADATA_URI", help="The URI of a metadata server")
@click.option("--database-type", type=click.Choice(SUPPORTED_DB_TYPES + ["api"]), envvar="ATRIUMDB_DATABASE_TYPE",
              help="The type of metadata database supporting the dataset.")
@click.option("--endpoint-url", type=str, envvar="ATRIUMDB_ENDPOINT_URL",
              help="The endpoint to connect to for a remote AtriumDB server")
@click.option("--api-token", type=str, envvar="ATRIUMDB_API_TOKEN",
              help="A token to authorize api access.")
@click.pass_context
def cli(ctx, dataset_location, metadata_uri, database_type, endpoint_url, api_token):
    ctx.ensure_object(dict)
    ctx.obj["endpoint_url"] = endpoint_url
    ctx.obj["api_token"] = api_token
    ctx.obj["dataset_location"] = dataset_location
    ctx.obj["metadata_uri"] = metadata_uri
    ctx.obj["database_type"] = database_type


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
@click.option("--endpoint-url-out", type=str, help="The endpoint to connect to for a remote AtriumDB server",
              envvar='ATRIUMDB_ENDPOINT_URL_OUT')
@click.option("--by-patient", type=bool, default=False, help="Whether or not to include patient mapping")
def export(ctx, export_format, packaging_type, cohort_file, measure_ids, measures, device_ids, devices, patient_ids,
           mrns, start_time, end_time, dataset_location_out, metadata_uri_out, endpoint_url_out, by_patient):
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

    from_sdk = get_sdk_from_cli_params(dataset_location, metadata_uri)

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
        to_sdk = get_sdk_from_cli_params(dataset_location_out, metadata_uri_out)

        transfer_data(from_sdk=from_sdk, to_sdk=to_sdk, measure_id_list=measure_ids, device_id_list=device_ids,
                      patient_id_list=patient_ids, mrn_list=mrns, start=start_time, end=end_time)

    else:
        assert dataset_location_out is not None, "dataset-location-out option or ATRIUMDB_EXPORT_DATASET_LOCATION " \
                                                 "envvar must be specified."

        dataset_dir = Path(dataset_location_out) / f"{export_format}_export"
        export_dataset(sdk=from_sdk, directory=dataset_dir, data_format=export_format, device_id_list=device_ids,
                       patient_id_list=patient_ids, mrn_list=mrns, start=start_time, end=end_time,
                       by_patient=by_patient, measure_id_list=measure_ids)


def get_sdk_from_cli_params(dataset_location, metadata_uri):
    connection_params = None if metadata_uri is None else parse_metadata_uri(metadata_uri)
    metadata_connection_type = None if connection_params is None else connection_params['sqltype']
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=metadata_connection_type,
                    connection_params=connection_params)
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
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]

    implemented_formats = ["csv", "parquet"]

    if import_format not in implemented_formats:
        raise NotImplementedError(f"Only {implemented_formats} formats are currently supported for import")

    if dataset_location_in is None:
        raise ValueError("dataset-location-in option or ATRIUMDB_IMPORT_DATASET_LOCATION envvar must be specified.")

    if not Path(dataset_location_in).is_dir():
        raise ValueError(f"{dataset_location_in} is not a valid directory.")

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri)

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
@click.option("--freq-hz", type=float, help="Filter measures by frequency in Hertz")
@click.option("--freq-nhz", type=int, help="Filter measures by frequency in Nanohertz")
@click.option("--source-id", type=int, help="Filter measures by source identifier")
def measure_ls(ctx, tag_match, name_match, unit, freq_hz, freq_nhz, source_id):
    endpoint_url = ctx.obj["endpoint_url"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri)
    result = sdk.search_measures(tag_match=tag_match, freq=freq_nhz, unit=unit, name_match=name_match)

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
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]

    sdk = get_sdk_from_cli_params(dataset_location, metadata_uri)
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
@click.option("--age-years-min", type=float, help="Filter patients by minimum age in years")
@click.option("--age-years-max", type=float, help="Filter patients by maximum age in years")
@click.option("--gender", type=str, help="Filter patients by gender")
@click.option("--source-id", type=str, help="Filter patients by source identifier")
@click.option("--first-seen", type=int, help="Filter patients by first seen timestamp in epoch time")
@click.option("--last-updated", type=int, help="Filter patients by last updated timestamp in epoch time")
def patient_ls(ctx, age_years_min, age_years_max, gender, source_id, first_seen, last_updated):
    endpoint_url = ctx.obj["endpoint_url"]
    dataset_location = ctx.obj["dataset_location"]
    metadata_uri = ctx.obj["metadata_uri"]

    click.echo("_LOGGER.infoing list of patients available in the configured dataset")
    click.echo(f"Endpoint URL: {endpoint_url}")
    click.echo(f"Dataset location: {dataset_location}")
    click.echo(f"Metadata URI: {metadata_uri}")
    if age_years_min:
        click.echo(f"Filter patients by minimum age in years: {age_years_min}")
    if age_years_max:
        click.echo(f"Filter patients by maximum age in years: {age_years_max}")
    if gender:
        click.echo(f"Filter patients by gender: {gender}")
    if source_id:
        click.echo(f"Filter patients by source identifier: {source_id}")
    if first_seen:
        click.echo(f"Filter patients by first seen timestamp in epoch time: {first_seen}")
    if last_updated:
        click.echo(f"Filter patients by last updated timestamp in epoch time: {last_updated}")


cli.add_command(export)
cli.add_command(import_)
cli.add_command(atriumdb_patient)
cli.add_command(atriumdb_measure)
cli.add_command(atriumdb_device)
