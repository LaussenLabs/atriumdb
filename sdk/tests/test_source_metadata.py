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
from tests.testing_framework import _test_for_both

DB_NAME = 'source_metadata'


def test_source_metadata():
    _test_for_both(DB_NAME, _test_source_metadata)


def _test_source_metadata(db_type, dataset_location, connection_params):
    atrium_sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    source_id = atrium_sdk.sql_handler.insert_source("Test Source", "A source description")
    assert source_id is not None
    selected_source = atrium_sdk.sql_handler.select_source(source_id=source_id)
    assert selected_source[1] == "Test Source"

    institution_id = atrium_sdk.sql_handler.insert_institution("Test Institution")
    assert institution_id is not None
    selected_institution = atrium_sdk.sql_handler.select_institution(institution_id=institution_id)
    assert selected_institution[1] == "Test Institution"

    unit_id = atrium_sdk.sql_handler.insert_unit(institution_id, "Test Unit", "General")
    assert unit_id is not None
    selected_unit = atrium_sdk.sql_handler.select_unit(unit_id=unit_id)
    assert selected_unit[2] == "Test Unit"

    bed_id = atrium_sdk.sql_handler.insert_bed(unit_id, "Bed 1")
    assert bed_id is not None
    selected_bed = atrium_sdk.sql_handler.select_bed(bed_id=bed_id)
    assert selected_bed[2] == "Bed 1"



