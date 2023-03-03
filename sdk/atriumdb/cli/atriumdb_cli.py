import click

from atriumdb.cli.sdk import get_sdk_from_env_vars, create_sdk_from_env_vars
from atriumdb.transfer.export_csv import export_csv_from_sdk
from atriumdb.transfer.import_csv import import_csv_to_sdk


cli_help_text = """
The atriumdb command is a command line interface for the Atrium database, 
allowing you to import and export data from the database.

The import-csv subcommand is used to import data into the database from a CSV file, while the export-csv subcommand 
is used to export data from the database to a CSV file. 

The SDK data is defined by the following environment variables:

ATRIUM_SDK_DATASET_LOCATION: This environment variable specifies the directory location where the Atrium SDK data will 
be stored.

ATRIUM_SDK_CONNECTION_TYPE: This environment variable specifies the type of metadata connection to be used for the 
Atrium SDK. The possible values are "sqlite", "mariadb", or "mysql".

ATRIUM_SDK_METADATA_HOST specifies the hostname or IP address of the metadata database server.

ATRIUM_SDK_METADATA_USER specifies the username used to connect to the metadata database.

ATRIUM_SDK_METADATA_PASSWORD specifies the password for the user specified by ATRIUM_SDK_METADATA_USER.

ATRIUM_SDK_METADATA_DATABASE specifies the name of the metadata database.

ATRIUM_SDK_METADATA_PORT specifies the port number used to connect to the metadata database.

For more information on how to use these subcommands 
and the Atrium database, please visit the documentation at https://atriumdb.sickkids.ca/docs/.
"""

@click.group(help=cli_help_text)
def cli():
    pass


@click.command()
def convert():
    pass


@click.command()
def transfer():
    pass


@click.command()
@click.option("--input-file", type=click.Path(exists=True), required=True, help="Path to the input CSV file")
def import_csv(input_file):
    sdk = create_sdk_from_env_vars()
    measure_id, device_id, start_time, end_time = import_csv_to_sdk(sdk, input_file)
    click.echo(f"Data imported successfully. measure_id: {measure_id}, device_id: {device_id}, "
               f"start_time: {start_time}, end_time: {end_time}")


@click.command()
@click.option("--output-file", type=click.Path(), required=True, help="Path to the output CSV file")
@click.option("--measure-id", type=int, required=True, help="Measure ID to export data for")
@click.option("--start-time", type=int, required=True, help="Start time of the data to export (nanosecond epoch)")
@click.option("--end-time", type=int, required=True, help="End time of the data to export (nanosecond epoch)")
@click.option("--device-id", type=int, help="Device ID to export data for")
def export_csv(output_file, measure_id, start_time, end_time, device_id):
    sdk = get_sdk_from_env_vars()
    export_csv_from_sdk(sdk, output_file, measure_id, start_time, end_time, device_id)
    click.echo(f"Data exported successfully to {output_file}")


@click.command()
@click.option("--tag-match", type=str, help="Tag to search for in measures")
@click.option("--freq-nhz", type=int, help="Frequency to search for in measures")
@click.option("--unit", type=str, help="Unit to search for in measures")
@click.option("--name-match", type=str, help="Name to search for in measures")
def measures(tag_match, freq_nhz, unit, name_match):
    sdk = get_sdk_from_env_vars()
    result = sdk.search_measures(tag_match=tag_match, freq_nhz=freq_nhz, unit=unit, name_match=name_match)
    click.echo("Measures:")
    for measure_id, measure_info in result.items():
        click.echo(f"Measure ID: {measure_id}")
        for key, value in measure_info.items():
            click.echo(f"    {key}: {value}")


@click.command()
@click.option("--tag-match", type=str, help="Tag to search for in devices")
@click.option("--name-match", type=str, help="Name to search for in devices")
def devices(tag_match, name_match):
    sdk = get_sdk_from_env_vars()
    result = sdk.search_devices(tag_match=tag_match, name_match=name_match)
    click.echo("Devices:")
    for device_id, device_info in result.items():
        click.echo(f"Device ID: {device_id}")
        for key, value in device_info.items():
            click.echo(f"    {key}: {value}")



cli.add_command(import_csv)
cli.add_command(export_csv)
cli.add_command(transfer)
cli.add_command(convert)
cli.add_command(measures)
cli.add_command(devices)
