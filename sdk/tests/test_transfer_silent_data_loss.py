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
"""Regression tests for silent ``transfer_data`` data-loss defects.

Both are silent: the transfer reports success, the destination is smaller or wronger
than the source, and nothing in the logs says which rows went missing. Silence is what
makes them worth a test -- a loud failure would have been caught by the first person to
hit it.

1. ``test_sparse_block_spanning_ranges_transfers_every_sample``

   A definition splits a device's data into time ranges at the gap tolerance. Sparse
   data puts many samples in ONE physical block, so a single block can span several of
   those ranges. Only the first range's worth of samples arrives.

   This is the aperiodic case specifically: a waveform at any real rate fills blocks
   long before it can straddle a gap-tolerance boundary, so the numeric waveform path
   this repository has always tested cannot reach the bug. The dataset below is the
   minimum that does -- eight samples, one write, one block, three validated ranges.

2. ``test_colliding_source_patient_ids_stay_distinct``

   Two independently-ingested sources each number their patients from 1. Transferring
   both into one destination with ``deidentify=False`` merges two different real
   people -- different MRNs -- into a single destination patient row. MRN is the
   identity that survives across sources; the source-local integer id is not.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_transfer_silent_data_loss.py -q -rx
"""
import shutil

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.transfer.adb.dataset import transfer_data
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-transfer-silent-loss'

SEC = 10 ** 9
GAP_TOLERANCE_NS = 5 * 60 * SEC

# Offsets in seconds from an arbitrary epoch. The first six sit inside one gap
# tolerance of each other; the last two are isolated, so validation yields three
# ranges while one write still produces a single block.
SPARSE_OFFSETS_S = [0, 3, 7, 21, 22, 100, 1_000, 1_000_000]
BASE_TIME_NS = 1_753_000_000_123_456_789


def _sibling(db_type, dataset_location, connection_params, suffix):
    """A second (or third) dataset alongside ``dataset_location``.

    ``create_sibling_sdk`` in the framework hard-codes one suffix and mutates the
    caller's ``connection_params``; these tests need up to three datasets at once, so
    they take a copy and name it explicitly.
    """
    location = f"{dataset_location}_{suffix}"
    shutil.rmtree(location, ignore_errors=True)
    params = dict(connection_params) if connection_params else connection_params
    if db_type in ('mysql', 'mariadb'):
        params['database'] = f"{connection_params['database']}-{suffix}"
        from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
        handler = MariaDBHandler(params['host'], params['user'], params['password'],
                                 params['database'], params['port'])
        handler.maria_connect_no_db().cursor().execute(
            f"DROP DATABASE IF EXISTS `{params['database']}`")
    return AtriumSDK.create_dataset(
        dataset_location=location, database_type=db_type, connection_params=params)


@pytest.mark.parametrize("backend", parametrized_backends())
def test_sparse_block_spanning_ranges_transfers_every_sample(backend):
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME, backend)
    source = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = source.insert_measure(
        "SpotO2", 1, freq_units="Hz", units="%", signal_kind="sample", value_type="numeric")
    device_id = source.insert_device("dev_sparse")

    times = BASE_TIME_NS + np.array(SPARSE_OFFSETS_S, dtype=np.int64) * SEC
    values = np.arange(len(SPARSE_OFFSETS_S), dtype=np.float64)
    source.write_time_value_pairs(measure_id, device_id, times, values)

    definition = DatasetDefinition(measures=["SpotO2"], device_ids={device_id: "all"})
    definition.validate(sdk=source, gap_tolerance=GAP_TOLERANCE_NS)

    # The bug needs both halves of its setup to actually hold, and both are properties
    # of the data rather than of the assertion -- so check them, or a future change to
    # block sizing or tolerance defaults would turn this into a test that passes
    # without exercising anything.
    ranges = definition.validated_data_dict['sources']['device_ids'][device_id]
    assert len(ranges) > 1, "definition must split into several ranges"
    headers, _, _ = source.get_data(measure_id, int(times[0]) - SEC, int(times[-1]) + SEC,
                                    device_id=device_id, time_type=1, sort=True)
    assert len(headers) == 1, "all samples must live in ONE block for the bug to appear"

    destination = _sibling(db_type, dataset_location, connection_params, "dest")
    transfer_data(source, destination, definition, export_format="tsc", deidentify=False)

    dest_measure_id = destination.get_measure_id("SpotO2", freq=1, freq_units="Hz", units="%")
    dest_device_id = destination.get_device_id("dev_sparse")
    _, dest_times, dest_values = destination.get_data(
        dest_measure_id, int(times[0]) - SEC, int(times[-1]) + SEC,
        device_id=dest_device_id, time_type=1, sort=True)

    missing = sorted(set(times.tolist()) - set(dest_times.tolist()))
    assert not missing, (
        f"{len(missing)} of {len(times)} samples were dropped by the transfer, silently: "
        f"offsets {[(t - BASE_TIME_NS) // SEC for t in missing]} s")
    np.testing.assert_array_equal(dest_values, values)


@pytest.mark.parametrize("backend", parametrized_backends())
def test_colliding_source_patient_ids_stay_distinct(backend):
    db_type, dataset_location, connection_params = prepare_backend(DB_NAME, backend)
    source_a = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    source_b = _sibling(db_type, dataset_location, connection_params, "srcb")
    destination = _sibling(db_type, dataset_location, connection_params, "dest2")

    end_time_n = 60 * SEC
    for sdk, mrn, name, device_tag in ((source_a, "MRN-AAA", "Alice", "dev_a"),
                                       (source_b, "MRN-BBB", "Bob", "dev_b")):
        measure_id = sdk.insert_measure("hr", 1, freq_units="Hz", units="bpm")
        device_id = sdk.insert_device(device_tag)
        patient_id = sdk.insert_patient(mrn=mrn, first_name=name)
        sdk.insert_device_patient_data([(device_id, patient_id, 0, end_time_n)])
        sdk.write_time_value_pairs(
            measure_id, device_id,
            np.arange(60, dtype=np.int64) * SEC,
            np.arange(60, dtype=np.float64))

    # The precondition: both sources numbered their first patient identically. If a
    # future change to id allocation breaks this, the test proves nothing.
    assert list(source_a.get_all_patients()) == list(source_b.get_all_patients()), \
        "both sources must use the same local patient ids for this collision to exist"

    for sdk, device_tag in ((source_a, "dev_a"), (source_b, "dev_b")):
        definition = DatasetDefinition(
            measures=["hr"], device_ids={sdk.get_device_id(device_tag): "all"})
        transfer_data(sdk, destination, definition, export_format="tsc", deidentify=False,
                      patient_info_to_transfer="all")

    patients = destination.get_all_patients()
    mrns = sorted(str(info['mrn']) for info in patients.values())
    assert mrns == ["MRN-AAA", "MRN-BBB"], (
        f"two different patients were merged into {len(patients)} destination record(s): "
        f"{mrns}. Source-local patient ids collided and MRN was not used to tell them apart")
