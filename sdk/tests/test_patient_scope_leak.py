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
"""Regression coverage for patient-scoped ``get_data`` reads.

A bed records continuously while its occupant changes. ``device_patient`` maps the device
to Alice for the first ten seconds and to Bob for the second ten. One physical block spans
the handover, so each patient query must trim that block to its own samples.

The cause is two patient-scoped read paths that disagree about what "this patient's data"
means:

* ``select_intervals(patient_id=)`` clips exactly, to
  ``max(start, encounter_start), min(end, encounter_end)``.
* ``select_blocks(patient_id=)`` selects blocks *overlapping* the encounter window and
  returns them whole. Its later filter only discards blocks that miss the caller's
  requested range; nothing ever trims a block to the encounter.

The interval index already reports the right span. The read must do the same, even when a
block crossing a handover contains samples for both occupants.

This is why it matters beyond correctness: patient-scoped reads are the mechanism by which
a cohort is extracted for people who are only entitled to that cohort, so the surplus is
another patient's identifiable physiological record.

The implementation keeps the physical-block overlap query and clips decoded samples to the
device-patient encounter. A patient query must not return another patient's samples.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \\
        atriumdb-test:latest python -m pytest \\
        /atriumdb/sdk/tests/test_patient_scope_leak.py -q -rx
"""
import numpy as np
import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import parametrized_backends, prepare_backend

DB_NAME = 'atrium-patient-scope'

SEC = 10 ** 9
HANDOVER_S = 10
TOTAL_S = 20


def _bed_with_two_occupants(db_type, dataset_location, connection_params):
    """One device, one continuous block, two patients in sequence."""
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    measure_id = sdk.insert_measure('hr', 1, freq_units='Hz', units='bpm')
    device_id = sdk.insert_device('bed_1')

    # A single write, so the samples land in one block that straddles the handover.
    times = np.arange(TOTAL_S, dtype=np.int64) * SEC
    values = np.arange(100, 100 + TOTAL_S, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values)

    alice = sdk.insert_patient(mrn='MRN-ALICE', first_name='Alice')
    bob = sdk.insert_patient(mrn='MRN-BOB', first_name='Bob')
    sdk.insert_device_patient_data([
        (device_id, alice, 0, HANDOVER_S * SEC),
        (device_id, bob, HANDOVER_S * SEC, TOTAL_S * SEC),
    ])
    return sdk, measure_id, alice, bob


@pytest.mark.parametrize("backend", parametrized_backends())
def test_patient_query_returns_only_that_patients_samples(backend):
    sdk, measure_id, alice, bob = _bed_with_two_occupants(
        *prepare_backend(DB_NAME, backend))

    _, alice_times, alice_values = sdk.get_data(measure_id, 0, TOTAL_S * SEC, patient_id=alice)
    _, bob_times, bob_values = sdk.get_data(measure_id, 0, TOTAL_S * SEC, patient_id=bob)

    alice_seconds = (alice_times // SEC).tolist()
    bob_seconds = (bob_times // SEC).tolist()

    assert alice_seconds == list(range(0, HANDOVER_S)), (
        f"Alice occupied the bed for [0,{HANDOVER_S}) s but her query returned seconds "
        f"{alice_seconds} -- the surplus is Bob's data")
    assert bob_seconds == list(range(HANDOVER_S, TOTAL_S)), (
        f"Bob occupied the bed for [{HANDOVER_S},{TOTAL_S}) s but his query returned "
        f"seconds {bob_seconds} -- the surplus is Alice's data")

    # Stated the other way round, because this is the property that actually matters.
    assert not set(alice_values.tolist()) & set(bob_values.tolist()), \
        "no sample may be returned for both patients"

    # Device queries remain physical-stream reads; only a patient selector applies
    # an encounter boundary.
    _, device_times, device_values = sdk.get_data(
        measure_id, 0, TOTAL_S * SEC, device_id=sdk.get_device_id('bed_1'))
    assert (device_times // SEC).tolist() == list(range(TOTAL_S))
    assert device_values.tolist() == list(range(100, 100 + TOTAL_S))
    sdk.close()


@pytest.mark.parametrize("backend", parametrized_backends())
def test_patient_intervals_and_reads_agree(backend):
    """The interval index already answers correctly. The read should match it."""
    sdk, measure_id, alice, _bob = _bed_with_two_occupants(
        *prepare_backend(DB_NAME + '_agree', backend))

    intervals = sdk.get_interval_array(measure_id, patient_id=alice,
                                       start=0, end=TOTAL_S * SEC)
    _, times, _values = sdk.get_data(measure_id, 0, TOTAL_S * SEC, patient_id=alice)

    assert intervals[-1][1] <= HANDOVER_S * SEC, \
        "precondition: the interval index is expected to clip to the encounter"
    assert int(times.max()) < int(intervals[-1][1]), (
        f"the read returned data up to {int(times.max())} ns, past the end of the "
        f"patient's own interval index at {int(intervals[-1][1])} ns")
    sdk.close()
