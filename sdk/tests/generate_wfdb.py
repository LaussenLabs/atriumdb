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

import wfdb
from pathlib import Path


DEFAULT_WFDB_DATA_DIR = Path(__file__).parent / 'wfdb_data'
DEFAULT_DATASET_NAME = 'mitdb'


def get_records(dataset_name=None, physical=True):
    dataset_name = DEFAULT_DATASET_NAME if dataset_name is None else dataset_name
    dataset_dir_path = DEFAULT_WFDB_DATA_DIR / dataset_name

    if not dataset_dir_path.is_dir():
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(dataset_name, str(dataset_dir_path))

    for record_name in wfdb.get_record_list(dataset_name):
        record = wfdb.rdrecord(str(dataset_dir_path / record_name), physical=physical)
        annotation = wfdb.rdann(str(dataset_dir_path / record_name), 'atr', summarize_labels=True, return_label_elements=['description'])
        yield (record, annotation)
