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

from atriumdb import AtriumSDK
from atriumdb.transfer.formats.import_data import import_data_to_sdk
from tests.testing_framework import _test_for_both
from atriumdb.transfer.formats.export_data import export_data_from_sdk

from pathlib import Path
import pandas as pd

DB_NAME = 'import_export'


def test_import_export():
    _test_for_both(DB_NAME, _test_import_export)


def _test_import_export(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Import data from a CSV file
    filename = Path(__file__).parent / "example_data" / "import_data_1.formats"
    measure_id, device_id, start_time, end_time = import_data_to_sdk(sdk, filename, )

    # Export data to a CSV file
    export_filename = Path(__file__).parent / "example_data" / "export_data_1.formats"
    export_data_from_sdk(sdk, export_filename, measure_id, start_time, end_time, device_id=device_id)

    # Check if the imported and exported data are the same
    imported_df = pd.read_csv(filename)
    exported_df = pd.read_csv(export_filename)

    print()
    print(imported_df)
    print()
    print(exported_df)

    assert imported_df.equals(exported_df), "Imported and exported data are not the same"



