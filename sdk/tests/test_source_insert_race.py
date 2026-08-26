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
"""
The get-or-insert contract of insert_measure / insert_device / get_or_insert_*.

Both inserts end in an ``INSERT [OR] IGNORE``. When a concurrent writer created the
same measure/device first, that statement matches nothing and the driver's
``lastrowid`` is empty -- ``None`` on MariaDB, ``0`` on SQLite (a fresh connection's
untouched value). The ``0`` used to slip past an ``is None`` guard, get returned as a
measure_id, AND get cached in ``_measure_ids`` / ``_measures``, so ``get_measure_id``
answered ``0`` for the rest of that SDK object's life and re-fetching could not recover.

Two kinds of coverage here, because they fail differently:

  * a DETERMINISTIC reproduction of the losing interleave (a second SDK object commits
    the row from inside the loser's own handler call). It runs on both backends, in the
    fast inner loop, and asserts the raw handler really did hand back an unusable id --
    so the test cannot pass vacuously if the interleave stops being reproduced.
  * a REAL multi-process race, several processes released from a barrier onto the same
    brand-new tag, asserting every process got the same real id and exactly one row
    exists.

SQLite plus (when configured) MariaDB:
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_source_insert_race.py -q
"""
import multiprocessing as mp
import shutil
import tempfile
import time

import pytest

from atriumdb import AtriumSDK
from tests.testing_framework import parametrized_backends, prepare_backend

HZ = "Hz"
TAG = "hr"
UNITS = "bpm"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sqlite_dataset():
    """A throwaway SQLite dataset, for the tests that have no backend dimension."""
    loc = tempfile.mkdtemp(prefix="atrium_race_")
    shutil.rmtree(loc, ignore_errors=True)
    sdk = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    try:
        yield sdk, loc
    finally:
        sdk.close()
        shutil.rmtree(loc, ignore_errors=True)


def _open(dataset_location, db_type, connection_params):
    return AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                     connection_params=connection_params)


def _lose_measure_race(loser_sdk, winner_action):
    """Make ``loser_sdk``'s next insert_measure lose the race.

    The winner commits the row from INSIDE the loser's handler call -- i.e. after the
    loser's own get_measure_id check has already missed -- which is exactly the window
    the bug lives in. The real handler still runs afterwards, so the ``lastrowid`` the
    loser sees is the driver's genuine answer, not a stubbed one. Its value is recorded
    in ``raw`` so a test can assert the interleave was actually reproduced.
    """
    real_insert = loser_sdk.sql_handler.insert_measure
    raw = {}

    def racing_insert(*args, **kwargs):
        winner_action()
        raw['value'] = real_insert(*args, **kwargs)
        return raw['value']

    loser_sdk.sql_handler.insert_measure = racing_insert
    return raw


def _lose_device_race(loser_sdk, winner_action):
    """insert_device twin of :func:`_lose_measure_race`."""
    real_insert = loser_sdk.sql_handler.insert_device
    raw = {}

    def racing_insert(*args, **kwargs):
        winner_action()
        raw['value'] = real_insert(*args, **kwargs)
        return raw['value']

    loser_sdk.sql_handler.insert_device = racing_insert
    return raw


def _assert_unusable(raw, what):
    """The interleave is only reproduced if the driver handed back nothing usable."""
    value = raw.get('value', 'NOT CALLED')
    assert value is None or (isinstance(value, int) and value <= 0), (
        f"the {what} race was not reproduced: the handler returned {value!r}, which is a "
        f"usable id, so this test is no longer exercising the lost-race branch")


# --------------------------------------------------------------------------- #
# 1. Deterministic lost race -- both backends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("db_type", parametrized_backends())
def test_insert_measure_losing_race_returns_the_winners_id(db_type):
    db_type, dataset_location, connection_params = prepare_backend('insert_race_measure', db_type)
    AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=db_type,
                             connection_params=connection_params)
    loser = _open(dataset_location, db_type, connection_params)
    winner = _open(dataset_location, db_type, connection_params)

    winner_id = {}
    raw = _lose_measure_race(
        loser,
        lambda: winner_id.setdefault(
            'id', winner.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)))

    got = loser.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)

    _assert_unusable(raw, "insert_measure")
    assert got == winner_id['id']
    assert isinstance(got, int) and got >= 1

    # One row, not two: INSERT IGNORE did its job.
    assert len(loser.get_all_measures()) == 1


@pytest.mark.parametrize("db_type", parametrized_backends())
def test_insert_device_losing_race_returns_the_winners_id(db_type):
    db_type, dataset_location, connection_params = prepare_backend('insert_race_device', db_type)
    AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=db_type,
                             connection_params=connection_params)
    loser = _open(dataset_location, db_type, connection_params)
    winner = _open(dataset_location, db_type, connection_params)

    winner_id = {}
    raw = _lose_device_race(
        loser,
        lambda: winner_id.setdefault('id', winner.insert_device(device_tag="monitor_a1")))

    got = loser.insert_device(device_tag="monitor_a1")

    _assert_unusable(raw, "insert_device")
    assert got == winner_id['id']
    assert isinstance(got, int) and got >= 1
    assert len(loser.get_all_devices()) == 1


# --------------------------------------------------------------------------- #
# 2. The cache-poisoning half of the defect (SQLite-specific in origin, checked
#    on both backends because the fix is backend-agnostic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("db_type", parametrized_backends())
def test_losing_race_does_not_poison_the_measure_cache(db_type):
    """The failure that made the defect unrecoverable: a bogus id cached under the
    (tag, freq, units) key, which get_measure_id consults BEFORE the database."""
    db_type, dataset_location, connection_params = prepare_backend('insert_race_cache', db_type)
    AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=db_type,
                             connection_params=connection_params)
    loser = _open(dataset_location, db_type, connection_params)
    winner = _open(dataset_location, db_type, connection_params)

    winner_id = {}
    raw = _lose_measure_race(
        loser,
        lambda: winner_id.setdefault(
            'id', winner.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)))

    got = loser.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)
    _assert_unusable(raw, "insert_measure")

    real_id = winner_id['id']
    assert got == real_id

    # Nothing unusable anywhere in either cache.
    assert all(cached >= 1 for cached in loser._measure_ids.values())
    assert 0 not in loser._measures
    assert None not in loser._measures

    # The re-fetch that used to return the poisoned 0.
    assert loser.get_measure_id(TAG, freq=1, freq_units=HZ, units=UNITS) == real_id
    info = loser.get_measure_info(real_id)
    assert info is not None and info['tag'] == TAG and info['unit'] == UNITS


def test_losing_race_applies_requested_kind_metadata(sqlite_dataset):
    """The winner created the row without our signal_kind / value_type. The loser must
    still classify it -- otherwise an ingest pipeline that lost one race leaves a
    'waveform' + string measure behind, the one combination windowing cannot iterate."""
    loser, loc = sqlite_dataset
    winner = AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite")

    winner_id = {}
    raw = _lose_measure_race(
        loser,
        lambda: winner_id.setdefault(
            'id', winner.insert_measure(measure_tag="alarm_text", freq=1, freq_units=HZ,
                                        units="string")))

    got = loser.insert_measure(measure_tag="alarm_text", freq=1, freq_units=HZ, units="string",
                               signal_kind="event", value_type="string")
    _assert_unusable(raw, "insert_measure")
    assert got == winner_id['id']
    assert (loser.get_measure_info(got)["signal_kind"], loser.get_measure_info(got)["value_type"]) == ("event", "string")

    # Persisted, not merely cached in the loser.
    fresh = AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite")
    assert (fresh.get_measure_info(got)["signal_kind"], fresh.get_measure_info(got)["value_type"]) == ("event", "string")


# --------------------------------------------------------------------------- #
# 3. Unresolvable -- raise, never return a falsy id
# --------------------------------------------------------------------------- #
def test_insert_measure_raises_when_no_id_can_be_obtained(sqlite_dataset):
    sdk, _loc = sqlite_dataset
    # An insert that reports "nothing inserted" and really did insert nothing: the
    # re-resolve finds no row either, which is the only case left where there is no
    # honest answer to return.
    sdk.sql_handler.insert_measure = lambda *args, **kwargs: 0

    with pytest.raises(RuntimeError, match="insert_measure could not obtain a measure_id"):
        sdk.insert_measure(measure_tag="ghost", freq=1, freq_units=HZ, units=UNITS)

    # And nothing was cached on the way out.
    assert all(cached >= 1 for cached in sdk._measure_ids.values())
    assert 0 not in sdk._measures


def test_insert_device_raises_when_no_id_can_be_obtained(sqlite_dataset):
    sdk, _loc = sqlite_dataset
    sdk.sql_handler.insert_device = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="insert_device could not obtain a device_id"):
        sdk.insert_device(device_tag="ghost_device")


# --------------------------------------------------------------------------- #
# 4. get_or_insert_measure / get_or_insert_device (R2)
# --------------------------------------------------------------------------- #
def test_get_or_insert_measure_creates_then_returns_the_same_id(sqlite_dataset):
    sdk, _loc = sqlite_dataset
    first = sdk.get_or_insert_measure(TAG, freq=1, freq_units=HZ, units=UNITS)
    assert isinstance(first, int) and first >= 1

    for _ in range(3):
        assert sdk.get_or_insert_measure(TAG, freq=1, freq_units=HZ, units=UNITS) == first

    assert sdk.get_measure_id(TAG, freq=1, freq_units=HZ, units=UNITS) == first
    assert len(sdk.get_all_measures()) == 1


def test_get_or_insert_device_creates_then_returns_the_same_id(sqlite_dataset):
    sdk, _loc = sqlite_dataset
    first = sdk.get_or_insert_device("monitor_a1")
    assert isinstance(first, int) and first >= 1

    for _ in range(3):
        assert sdk.get_or_insert_device("monitor_a1") == first

    assert sdk.get_device_id("monitor_a1") == first
    assert len(sdk.get_all_devices()) == 1


def test_get_or_insert_measure_finds_a_measure_made_by_insert_measure(sqlite_dataset):
    """The two are one get-or-insert, not two registries."""
    sdk, _loc = sqlite_dataset
    made = sdk.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)
    assert sdk.get_or_insert_measure(TAG, freq=1, freq_units=HZ, units=UNITS) == made


def test_get_or_insert_measure_classifies_an_existing_unclassified_measure(sqlite_dataset):
    """An earlier run created the measure with no kind metadata; this run declares it."""
    sdk, _loc = sqlite_dataset
    made = sdk.insert_measure(measure_tag="alarm_text", freq=1, freq_units=HZ, units="string")
    assert (sdk.get_measure_info(made)["signal_kind"], sdk.get_measure_info(made)["value_type"]) == ("waveform", "numeric")

    same = sdk.get_or_insert_measure("alarm_text", freq=1, freq_units=HZ, units="string",
                                     signal_kind="event", value_type="string")
    assert same == made
    assert (sdk.get_measure_info(made)["signal_kind"], sdk.get_measure_info(made)["value_type"]) == ("event", "string")


def test_get_or_insert_measure_survives_a_lost_race(sqlite_dataset):
    """The wrapper inherits R1's guarantee -- that is the point of shipping them together."""
    loser, loc = sqlite_dataset
    winner = AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite")

    winner_id = {}
    raw = _lose_measure_race(
        loser,
        lambda: winner_id.setdefault(
            'id', winner.insert_measure(measure_tag=TAG, freq=1, freq_units=HZ, units=UNITS)))

    got = loser.get_or_insert_measure(TAG, freq=1, freq_units=HZ, units=UNITS)
    _assert_unusable(raw, "insert_measure")
    assert got == winner_id['id'] >= 1


# --------------------------------------------------------------------------- #
# 5. Real multi-process race
# --------------------------------------------------------------------------- #
def _race_worker(dataset_location, db_type, connection_params, start_at, tag):
    """Open an SDK, wait for the shared release time, then get-or-insert one measure
    and one device. Returns the ids so the parent can check they all agree."""
    sdk = AtriumSDK(dataset_location=str(dataset_location), metadata_connection_type=db_type,
                    connection_params=connection_params)
    try:
        delay = start_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        measure_id = sdk.get_or_insert_measure(tag, freq=1, freq_units=HZ, units=UNITS)
        device_id = sdk.get_or_insert_device(f"dev_{tag}")
        return measure_id, device_id
    finally:
        sdk.close()


@pytest.mark.parametrize("db_type", parametrized_backends())
def test_concurrent_processes_all_get_the_same_real_id(db_type):
    """The real thing: N processes released together onto a brand-new tag. Every one of
    them must come back with the same id, and it must be a real one."""
    db_type, dataset_location, connection_params = prepare_backend('insert_race_procs', db_type)
    AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=db_type,
                             connection_params=connection_params)

    num_processes = 6
    tag = "raced_measure"
    # A wall-clock barrier: cheap, needs no shared primitive across the spawn boundary,
    # and lands every worker in the window within a few milliseconds of each other.
    start_at = time.monotonic() + 2.0

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_processes) as pool:
        results = pool.starmap(
            _race_worker,
            [(str(dataset_location), db_type, connection_params, start_at, tag)] * num_processes)

    measure_ids = {measure_id for measure_id, _ in results}
    device_ids = {device_id for _, device_id in results}

    assert len(measure_ids) == 1, f"processes disagreed about the measure_id: {results}"
    assert len(device_ids) == 1, f"processes disagreed about the device_id: {results}"
    assert measure_ids.pop() >= 1
    assert device_ids.pop() >= 1

    sdk = _open(dataset_location, db_type, connection_params)
    assert len(sdk.get_all_measures()) == 1
    assert len(sdk.get_all_devices()) == 1
    sdk.close()
