from atriumdb import AtriumSDK
from atriumdb.transfer.csv.import_csv import import_csv_to_sdk
from tests.testing_framework import _test_for_both
from atriumdb.transfer.csv.export_csv import export_csv_from_sdk

from pathlib import Path
import pandas as pd

DB_NAME = 'import_export'


def test_import_export():
    _test_for_both(DB_NAME, _test_import_export)


def _test_import_export(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Import data from a CSV file
    filename = Path(__file__).parent / "example_data" / "import_data_1.csv"
    measure_id, device_id, start_time, end_time = import_csv_to_sdk(sdk, filename)

    # Export data to a CSV file
    export_filename = Path(__file__).parent / "example_data" / "export_data_1.csv"
    export_csv_from_sdk(sdk, export_filename, measure_id, start_time, end_time, device_id=device_id)

    # Check if the imported and exported data are the same
    imported_df = pd.read_csv(filename)
    exported_df = pd.read_csv(export_filename)

    print()
    print(imported_df)
    print()
    print(exported_df)

    assert imported_df.equals(exported_df), "Imported and exported data are not the same"



