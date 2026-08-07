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

import os
import wfdb
from pathlib import Path

# The WFDB cache location can be redirected with the ATRIUMDB_WFDB_CACHE environment
# variable so that CI / container runs can mount a pre-populated (possibly read-only)
# cache instead of downloading from PhysioNet on every run.
_ENV_WFDB_CACHE = "ATRIUMDB_WFDB_CACHE"

DEFAULT_WFDB_DATA_DIR = Path(os.getenv(_ENV_WFDB_CACHE) or (Path(__file__).parent / 'wfdb_data'))
DEFAULT_DATASET_NAME = 'mitdb'

RECORDS_FILE_NAME = 'RECORDS'


def get_wfdb_cache_dir():
    """Return the root directory the WFDB test data is cached in.

    Re-read from the environment on each call so that tests (or a conftest) may set
    ATRIUMDB_WFDB_CACHE after this module has been imported.
    """
    env_value = os.getenv(_ENV_WFDB_CACHE)
    if env_value:
        return Path(env_value)
    return DEFAULT_WFDB_DATA_DIR


def _cached_record_list(dataset_dir_path):
    """Derive the record list from the local cache, without touching the network.

    Prefers a RECORDS file (written by us when we download) and otherwise enumerates
    the header files that are actually present. Returns a deterministically sorted
    list, or an empty list if nothing usable is cached.
    """
    records_file = dataset_dir_path / RECORDS_FILE_NAME
    if records_file.is_file():
        names = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
        # Only trust entries that really do have a header file on disk.
        names = [name for name in names if (dataset_dir_path / f"{name}.hea").is_file()]
        if names:
            return sorted(names)

    if not dataset_dir_path.is_dir():
        return []

    names = []
    for header_path in dataset_dir_path.rglob('*.hea'):
        relative = header_path.relative_to(dataset_dir_path).with_suffix('')
        names.append(relative.as_posix())
    return sorted(names)


def _download_dataset(dataset_name, dataset_dir_path):
    """Download a dataset into the cache, raising an actionable error on failure."""
    try:
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        record_names = list(wfdb.get_record_list(dataset_name))
        wfdb.dl_database(dataset_name, str(dataset_dir_path))
    except Exception as download_error:
        raise RuntimeError(
            f"The WFDB dataset '{dataset_name}' is not cached locally at "
            f"'{dataset_dir_path}' and it could not be downloaded "
            f"({type(download_error).__name__}: {download_error}).\n"
            f"To run these tests offline, populate that directory with the "
            f"'{dataset_name}' record files (the .hea/.dat/.atr files from "
            f"https://physionet.org/content/{dataset_name}/), or point the "
            f"{_ENV_WFDB_CACHE} environment variable at a directory that already "
            f"contains a '{dataset_name}' subdirectory with them."
        ) from download_error

    # Persist the record list so subsequent runs never need the network.
    try:
        (dataset_dir_path / RECORDS_FILE_NAME).write_text("\n".join(record_names) + "\n")
    except OSError:
        # A read-only cache is fine; _cached_record_list falls back to globbing.
        pass

    return sorted(record_names)


def get_record_names(dataset_name=None):
    """Return the sorted record names for `dataset_name`, downloading only if needed."""
    dataset_name = DEFAULT_DATASET_NAME if dataset_name is None else dataset_name
    dataset_dir_path = get_wfdb_cache_dir() / dataset_name

    record_names = _cached_record_list(dataset_dir_path)
    if record_names:
        return record_names

    return _download_dataset(dataset_name, dataset_dir_path)


def get_records(dataset_name=None, physical=True):
    dataset_name = DEFAULT_DATASET_NAME if dataset_name is None else dataset_name
    dataset_dir_path = get_wfdb_cache_dir() / dataset_name

    for record_name in get_record_names(dataset_name):
        record = wfdb.rdrecord(str(dataset_dir_path / record_name), physical=physical)
        annotation = wfdb.rdann(str(dataset_dir_path / record_name), 'atr', summarize_labels=True, return_label_elements=['description'])
        yield (record, annotation)
