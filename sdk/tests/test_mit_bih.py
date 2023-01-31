from atriumdb import AtriumSDK
import wfdb
import numpy as np
from pathlib import Path
import shutil

TSC_DATASET_DIR = Path(__file__).parent / 'mit_bih'

def test_mit_bih():
    # Define where on disk the dataset will sit.
    try:
        TSC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        sdk = AtriumSDK(dataset_location=str(TSC_DATASET_DIR))

        record_names = wfdb.get_record_list('mitdb')

        # populate database
        for n in record_names:

            record = wfdb.rdrecord(n, pn_dir="mitdb")

            # make a new device for each record and make the device tag the name of the record
            device_id = sdk.get_device_id(device_tag=record.record_name)
            if device_id is None:
                device_id = sdk.insert_device(device_tag=record.record_name)

            freq_nano = record.fs * 1_000_000_000

            time_arr = np.arange(record.sig_len, dtype=np.int64) * int(10 ** 9 // record.fs)

            # if there are multiple signals in one record split them into two different dataset entries
            if record.n_sig > 1:
                for i in range(len(record.sig_name)):

                    # if the measure tag has already been entered into the DB find the associated measure ID
                    measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])
                    if measure_id is None:
                        # if the measure, frequency pair is not in the DB create a new entry
                        measure_id = sdk.insert_measure(measure_tag=record.sig_name[i], freq_nhz=freq_nano, units=record.units[i])

                    # write data
                    sdk.write_data_easy(measure_id, device_id, time_arr, record.p_signal.T[i],
                                        freq_nano, scale_m=None, scale_b=None)

            # if there is only one signal in the input file insert it
            else:
                measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)
                if measure_id is None:
                    measure_id = sdk.insert_measure(measure_tag=record.sig_name, freq_nhz=freq_nano, units=record.units)

                sdk.write_data_easy(measure_id, device_id, time_arr, record.p_signal,
                                    freq_nano, scale_m=None, scale_b=None)


        # Check if all entries have been entered correctly
        for n in record_names:

            record = wfdb.rdrecord(n, pn_dir="mitdb")
            freq_nano = record.fs * 1_000_000_000
            time_arr = np.arange(record.sig_len, dtype=np.int64) * ((10 ** 9) // record.fs)
            device_id = sdk.get_device_id(device_tag=record.record_name)

            # If there are multiple signals in the record check both
            if record.n_sig > 1:
                for i in range(len(record.sig_name)):
                    measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])

                    _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)

                    # check that both the signal and time arrays from mitDB and atriumDB are equal
                    assert np.array_equal(record.p_signal.T[i], read_values) and np.array_equal(time_arr, read_times)

            # If there is only one signal in the record
            else:
                measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)

                _, read_times, read_values = sdk.get_data(measure_id, 0, 10 ** 18, device_id=device_id)

                assert np.array_equal(record.p_signal, read_values) and np.array_equal(time_arr, read_times)
    finally:
        shutil.rmtree(TSC_DATASET_DIR)

def reset_database(highest_level_dir):
    db_path = f"{highest_level_dir}/meta/index.db"
    tsc_path = f"{highest_level_dir}/tsc"

    Path(db_path).unlink(missing_ok=True)
    if Path(tsc_path).is_dir():
        shutil.rmtree(tsc_path)
