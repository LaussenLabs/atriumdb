import wfdb
from pathlib import Path


DEFAULT_WFDB_DATA_DIR = Path(__file__).parent / 'wfdb_data'
DEFAULT_DATASET_NAME = 'mitdb'


def get_records(dataset_name=None):
    dataset_name = DEFAULT_DATASET_NAME if dataset_name is None else dataset_name
    dataset_dir_path = DEFAULT_WFDB_DATA_DIR / dataset_name

    if not dataset_dir_path.is_dir():
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(dataset_name, str(dataset_dir_path))

    for record_name in wfdb.get_record_list(dataset_name):
        record = wfdb.rdrecord(str(dataset_dir_path / record_name))
        yield record
