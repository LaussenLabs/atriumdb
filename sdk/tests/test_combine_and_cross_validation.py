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
import tempfile
from math import comb

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition, combine_definitions, partition_dataset, cross_validate_dataset
from atriumdb.windowing.definition_combine import (
    _merge_measures,
    _merge_labels,
    _merge_source_dicts,
    _merge_validated_sources,
)
from atriumdb.windowing.cross_validation import (
    _generate_fold_combinations,
    _combine_fold_definitions,
)
from tests.test_mit_bih import write_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'combine-cv'
MAX_RECORDS = 10
SEED = 42


# ===================================================================
# _merge_measures
# ===================================================================

def test_merge_measures_dedup_strings():
    """Duplicate string measures are deduplicated."""
    result = _merge_measures([["ECG", "ABP"], ["ABP", "SpO2"]])
    assert result == ["ECG", "ABP", "SpO2"]


def test_merge_measures_dedup_dicts():
    """Duplicate dict measures with matching tag/freq/units are deduplicated."""
    m1 = [{"tag": "ECG", "freq_hz": 500, "units": "mV"}]
    m2 = [{"tag": "ECG", "freq_hz": 500, "units": "mV"}, {"tag": "ABP", "freq_hz": 125, "units": "mmHg"}]
    result = _merge_measures([m1, m2])
    assert len(result) == 2
    assert result[0] == {"tag": "ECG", "freq_hz": 500, "units": "mV"}
    assert result[1] == {"tag": "ABP", "freq_hz": 125, "units": "mmHg"}


def test_merge_measures_string_and_dict_kept_separate():
    """A string tag and a dict triplet for the same tag are distinct specifications."""
    result = _merge_measures([["ECG"], [{"tag": "ECG", "freq_hz": 500, "units": "mV"}]])
    assert len(result) == 2


def test_merge_measures_preserves_order():
    """Measures appear in the order they were first seen."""
    result = _merge_measures([["C", "B", "A"], ["D"]])
    assert result == ["C", "B", "A", "D"]


# ===================================================================
# _merge_labels
# ===================================================================

def test_merge_labels_dedup():
    result = _merge_labels([["sinus", "afib"], ["afib", "noise"]])
    assert result == ["sinus", "afib", "noise"]


def test_merge_labels_empty():
    result = _merge_labels([[], []])
    assert result == []


# ===================================================================
# _merge_source_dicts
# ===================================================================

def test_merge_source_dicts_disjoint():
    d1 = {1: [{"start": 100, "end": 200}]}
    d2 = {2: [{"start": 300, "end": 400}]}
    result = _merge_source_dicts([d1, d2])
    assert 1 in result and 2 in result


def test_merge_source_dicts_same_key_concat():
    d1 = {1: [{"start": 100, "end": 200}]}
    d2 = {1: [{"start": 300, "end": 400}]}
    result = _merge_source_dicts([d1, d2])
    assert len(result[1]) == 2


def test_merge_source_dicts_all_wins():
    """If either side is 'all', the result is 'all'."""
    d1 = {1: [{"start": 100, "end": 200}]}
    d2 = {1: "all"}
    result = _merge_source_dicts([d1, d2])
    assert result[1] == "all"


def test_merge_source_dicts_all_both_sides():
    result = _merge_source_dicts([{1: "all"}, {1: "all"}])
    assert result[1] == "all"


# ===================================================================
# _merge_validated_sources
# ===================================================================

def test_merge_validated_sources_union():
    """Overlapping intervals for the same device_patient_tuple are unioned."""
    s1 = {"device_patient_tuples": {(1, 10): [[0, 100], [200, 300]]}}
    s2 = {"device_patient_tuples": {(1, 10): [[50, 150]], (2, 20): [[0, 50]]}}
    result = _merge_validated_sources([s1, s2])

    dpt = result["device_patient_tuples"]
    assert (1, 10) in dpt
    assert (2, 20) in dpt

    # [0,100] ∪ [50,150] → [0,150], and [200,300] stays separate
    intervals = dpt[(1, 10)]
    assert len(intervals) == 2
    assert intervals[0][0] == 0 and intervals[0][1] == 150
    assert intervals[1][0] == 200 and intervals[1][1] == 300


# ===================================================================
# combine_definitions
# ===================================================================

def test_combine_definitions_basic_unvalidated():
    def_a = DatasetDefinition(measures=["ECG"], labels=["sinus"], patient_ids={1: "all"})
    def_b = DatasetDefinition(measures=["ABP"], labels=["afib"], patient_ids={2: "all"})
    combined = combine_definitions([def_a, def_b])

    assert "ECG" in combined.data_dict['measures']
    assert "ABP" in combined.data_dict['measures']
    assert "sinus" in combined.data_dict['labels']
    assert "afib" in combined.data_dict['labels']
    assert 1 in combined.data_dict['patient_ids']
    assert 2 in combined.data_dict['patient_ids']
    assert not combined.is_validated


def test_combine_definitions_dedup_measures():
    def_a = DatasetDefinition(measures=["ECG", "ABP"])
    def_b = DatasetDefinition(measures=["ABP", "SpO2"])
    combined = combine_definitions([def_a, def_b])
    assert combined.data_dict['measures'] == ["ECG", "ABP", "SpO2"]


def test_combine_definitions_overlapping_sources():
    def_a = DatasetDefinition(measures=["ECG"], patient_ids={1: [{"start": 0, "end": 100}]})
    def_b = DatasetDefinition(measures=["ECG"], patient_ids={1: [{"start": 200, "end": 300}]})
    combined = combine_definitions([def_a, def_b])
    assert len(combined.data_dict['patient_ids'][1]) == 2


def test_combine_definitions_all_wins():
    def_a = DatasetDefinition(measures=["ECG"], patient_ids={1: [{"start": 0, "end": 100}]})
    def_b = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    combined = combine_definitions([def_a, def_b])
    assert combined.data_dict['patient_ids'][1] == "all"


def test_combine_definitions_mixed_validation_warns():
    def_a = DatasetDefinition(measures=["ECG"])
    def_b = DatasetDefinition(measures=["ECG"])
    # Manually mark one as validated for testing
    def_b.is_validated = True
    def_b.validated_data_dict = {'measures': [], 'labels': [], 'sources': {}}
    with pytest.warns(UserWarning, match="mixed validation status"):
        combined = combine_definitions([def_a, def_b])
    assert not combined.is_validated
    assert "ECG" in combined.data_dict['measures']


def test_combine_definitions_fewer_than_two_raises():
    with pytest.raises(ValueError, match="At least two"):
        combine_definitions([DatasetDefinition(measures=["ECG"])])


def test_combine_definitions_instance_method():
    def_a = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    def_b = DatasetDefinition(measures=["ABP"], patient_ids={2: "all"})
    combined = def_a.combine(def_b)
    assert "ECG" in combined.data_dict['measures']
    assert "ABP" in combined.data_dict['measures']


def test_combine_definitions_multiple():
    defs = [
        DatasetDefinition(measures=["ECG"], patient_ids={1: "all"}),
        DatasetDefinition(measures=["ABP"], patient_ids={2: "all"}),
        DatasetDefinition(measures=["SpO2"], patient_ids={3: "all"}),
    ]
    combined = combine_definitions(defs)
    assert len(combined.data_dict['measures']) == 3
    assert len(combined.data_dict['patient_ids']) == 3


def test_combine_definitions_validated():
    """Two validated definitions combine their validated_data_dict."""
    def_a = DatasetDefinition(measures=["ECG"])
    def_b = DatasetDefinition(measures=["ABP"])
    # Simulate validated state
    for d, measure_id, measure_tag in [(def_a, 1, "ECG"), (def_b, 2, "ABP")]:
        d.is_validated = True
        d.validated_gap_tolerance = 1_000_000_000
        d.validated_data_dict = {
            'measures': [{'id': measure_id, 'tag': measure_tag, 'freq_nhz': 500_000_000_000, 'unit': 'mV'}],
            'labels': [],
            'sources': {
                'device_patient_tuples': {(1, measure_id): [[0, 100_000_000_000]]},
            },
        }
    combined = combine_definitions([def_a, def_b])
    assert combined.is_validated
    assert len(combined.validated_data_dict['measures']) == 2
    assert combined.validated_gap_tolerance == 1_000_000_000


# ===================================================================
# add_measure
# ===================================================================

def test_add_measure_simple():
    d = DatasetDefinition()
    d.add_measure("ECG")
    assert "ECG" in d.data_dict['measures']


def test_add_measure_with_freq_and_units():
    d = DatasetDefinition()
    d.add_measure("ECG", freq=500, units="mV")
    assert d.data_dict['measures'] == [{"tag": "ECG", "freq_hz": 500, "units": "mV"}]


def test_add_measure_freq_zero():
    """freq=0 is a valid frequency (e.g. aperiodic). It should not be silently dropped."""
    d = DatasetDefinition()
    d.add_measure("EVENT", freq=0, units="count")
    assert len(d.data_dict['measures']) == 1
    assert d.data_dict['measures'][0] == {"tag": "EVENT", "freq_hz": 0, "units": "count"}


def test_add_measure_only_freq_raises():
    d = DatasetDefinition()
    with pytest.raises(ValueError, match="Both freq and units"):
        d.add_measure("ECG", freq=500)


def test_add_measure_only_units_raises():
    d = DatasetDefinition()
    with pytest.raises(ValueError, match="Both freq and units"):
        d.add_measure("ECG", units="mV")


def test_add_measure_duplicate_warns():
    d = DatasetDefinition(measures=["ECG"])
    with pytest.warns(UserWarning, match="already present"):
        d.add_measure("ECG")
    assert d.data_dict['measures'] == ["ECG"]


def test_add_measure_invalidates():
    d = DatasetDefinition(measures=["ECG"])
    d.is_validated = True
    d.validated_data_dict = {'measures': [], 'labels': [], 'sources': {}}
    with pytest.warns(UserWarning, match="modified after validation"):
        d.add_measure("ABP")
    assert not d.is_validated


def test_add_measure_returns_self():
    d = DatasetDefinition()
    result = d.add_measure("ECG")
    assert result is d


# ===================================================================
# add_label
# ===================================================================

def test_add_label():
    d = DatasetDefinition()
    d.add_label("sinus")
    assert "sinus" in d.data_dict['labels']


def test_add_label_duplicate_raises():
    d = DatasetDefinition(labels=["sinus"])
    with pytest.raises(ValueError, match="already present"):
        d.add_label("sinus")


def test_add_label_invalidates():
    d = DatasetDefinition(labels=["sinus"])
    d.is_validated = True
    d.validated_data_dict = {'measures': [], 'labels': [], 'sources': {}}
    with pytest.warns(UserWarning, match="modified after validation"):
        d.add_label("afib")
    assert not d.is_validated


def test_add_label_returns_self():
    d = DatasetDefinition()
    assert d.add_label("sinus") is d


# ===================================================================
# add_region
# ===================================================================

def test_add_region_interval():
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(patient_id=1, start=100, end=200)
    assert d.data_dict['patient_ids'][1] == [{"start": 100, "end": 200}]


def test_add_region_event():
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(patient_id=1, time0=150, pre=50, post=50)
    assert d.data_dict['patient_ids'][1] == [{"time0": 150, "pre": 50, "post": 50}]


def test_add_region_all():
    """Omitting all time params requests all data for the source."""
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(patient_id=1)
    assert d.data_dict['patient_ids'][1] == "all"


def test_add_region_start_only():
    """start without end represents an open-ended region."""
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(patient_id=1, start=100)
    assert d.data_dict['patient_ids'][1] == [{"start": 100}]


def test_add_region_end_only():
    """end without start represents everything up to that point."""
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(patient_id=1, end=200)
    assert d.data_dict['patient_ids'][1] == [{"end": 200}]


def test_add_region_mixing_interval_event_raises():
    d = DatasetDefinition(measures=["ECG"])
    with pytest.raises(ValueError, match="Cannot specify both"):
        d.add_region(patient_id=1, start=100, time0=150, pre=50, post=50)


def test_add_region_incomplete_event_raises():
    d = DatasetDefinition(measures=["ECG"])
    with pytest.raises(ValueError, match="time0, pre, and post must all be"):
        d.add_region(patient_id=1, time0=150, pre=50)


def test_add_region_no_source_raises():
    d = DatasetDefinition(measures=["ECG"])
    with pytest.raises(ValueError, match="One of patient_id"):
        d.add_region(start=100, end=200)


def test_add_region_multiple_sources_raises():
    d = DatasetDefinition(measures=["ECG"])
    with pytest.raises(ValueError, match="Only one of"):
        d.add_region(patient_id=1, device_id=2, start=100, end=200)


def test_add_region_append_to_existing():
    d = DatasetDefinition(measures=["ECG"], patient_ids={1: [{"start": 0, "end": 50}]})
    d.add_region(patient_id=1, start=100, end=200)
    assert len(d.data_dict['patient_ids'][1]) == 2


def test_add_region_all_on_existing_specific_warns():
    d = DatasetDefinition(measures=["ECG"], patient_ids={1: [{"start": 0, "end": 50}]})
    with pytest.warns(UserWarning, match="Replacing with"):
        d.add_region(patient_id=1)
    assert d.data_dict['patient_ids'][1] == "all"


def test_add_region_specific_on_existing_all_warns():
    d = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    with pytest.warns(UserWarning, match="already requests all data"):
        d.add_region(patient_id=1, start=100, end=200)
    # Should remain "all"
    assert d.data_dict['patient_ids'][1] == "all"


def test_add_region_invalidates():
    d = DatasetDefinition(measures=["ECG"])
    d.is_validated = True
    d.validated_data_dict = {'measures': [], 'labels': [], 'sources': {}}
    with pytest.warns(UserWarning, match="modified after validation"):
        d.add_region(patient_id=1, start=100, end=200)
    assert not d.is_validated


def test_add_region_returns_self():
    d = DatasetDefinition(measures=["ECG"])
    assert d.add_region(patient_id=1, start=100, end=200) is d


def test_add_region_device_tag():
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(device_tag="monitor_1", start=100, end=200)
    assert "monitor_1" in d.data_dict['device_tags']


def test_add_region_mrn():
    d = DatasetDefinition(measures=["ECG"])
    d.add_region(mrn=12345, start=100, end=200)
    assert "12345" in d.data_dict['mrns']


# ===================================================================
# _generate_fold_combinations
# ===================================================================

def test_fold_combinations_5_1_1():
    """5 folds with 1 val and 1 test produces C(5,1)*C(4,1)=20 combos."""
    combos = _generate_fold_combinations(5, 1, 1)
    assert len(combos) == 20
    for c in combos:
        assert len(c['val']) == 1
        assert len(c['test']) == 1
        assert len(c['train']) == 3
        assert set(c['val']) | set(c['test']) | set(c['train']) == {0, 1, 2, 3, 4}


def test_fold_combinations_4_1_1():
    combos = _generate_fold_combinations(4, 1, 1)
    assert len(combos) == 12  # C(4,1)*C(3,1)


def test_fold_combinations_3_1_1():
    combos = _generate_fold_combinations(3, 1, 1)
    assert len(combos) == 6  # C(3,1)*C(2,1)


def test_fold_combinations_5_2_1():
    combos = _generate_fold_combinations(5, 2, 1)
    assert len(combos) == 30  # C(5,2)*C(3,1)
    for c in combos:
        assert len(c['val']) == 2
        assert len(c['test']) == 1
        assert len(c['train']) == 2


def test_fold_combinations_no_duplicates():
    combos = _generate_fold_combinations(5, 1, 1)
    keys = set()
    for c in combos:
        key = (c['val'], c['test'], c['train'])
        assert key not in keys, f"Duplicate combination found: {key}"
        keys.add(key)


def test_fold_combinations_formula():
    """Number of combinations matches math formula for various n, nv, nt."""
    for n in [3, 4, 5, 6]:
        for nv in [0, 1, 2]:
            for nt in [0, 1, 2]:
                if nv == 0 and nt == 0:
                    continue
                if n - nv - nt < 1:
                    continue
                expected = comb(n, nv) * comb(n - nv, nt)
                combos = _generate_fold_combinations(n, nv, nt)
                assert len(combos) == expected, (
                    f"n={n}, nv={nv}, nt={nt}: expected {expected}, got {len(combos)}")


def test_fold_combinations_zero_val():
    """n_val_folds=0 produces combinations with empty val and all folds split between test and train."""
    combos = _generate_fold_combinations(5, 0, 1)
    assert len(combos) == 5  # C(5,0)*C(5,1) = 5
    for c in combos:
        assert c['val'] == ()
        assert len(c['test']) == 1
        assert len(c['train']) == 4
        assert set(c['test']) | set(c['train']) == {0, 1, 2, 3, 4}


def test_fold_combinations_zero_test():
    """n_test_folds=0 produces combinations with empty test and all folds split between val and train."""
    combos = _generate_fold_combinations(5, 1, 0)
    assert len(combos) == 5  # C(5,1)*C(4,0) = 5
    for c in combos:
        assert c['test'] == ()
        assert len(c['val']) == 1
        assert len(c['train']) == 4
        assert set(c['val']) | set(c['train']) == {0, 1, 2, 3, 4}


# ===================================================================
# _combine_fold_definitions
# ===================================================================

def test_combine_fold_definitions_single():
    d = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    result = _combine_fold_definitions([d])
    assert result.data_dict['measures'] == ["ECG"]
    assert 1 in result.data_dict['patient_ids']
    assert result is not d  # should be a copy


def test_combine_fold_definitions_multiple():
    d1 = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    d2 = DatasetDefinition(measures=["ECG"], patient_ids={2: "all"})
    result = _combine_fold_definitions([d1, d2])
    assert 1 in result.data_dict['patient_ids']
    assert 2 in result.data_dict['patient_ids']


# ===================================================================
# SDK-dependent: combine_definitions with real data
# ===================================================================

def test_combine_definitions_with_sdk():
    _test_for_both(DB_NAME, _test_combine_definitions_with_sdk)


def _test_combine_definitions_with_sdk(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk.get_all_measures().values()]
    all_device_ids = list(sdk.get_all_devices().keys())

    # Split devices into two groups and make separate definitions
    mid = len(all_device_ids) // 2
    device_ids_a = {np.int64(did): "all" for did in all_device_ids[:mid]}
    device_ids_b = {np.int64(did): "all" for did in all_device_ids[mid:]}

    def_a = DatasetDefinition(measures=measures, device_ids=device_ids_a)
    def_b = DatasetDefinition(measures=measures, device_ids=device_ids_b)

    # Combine unvalidated
    combined = combine_definitions([def_a, def_b])
    assert not combined.is_validated
    assert len(combined.data_dict['device_ids']) == len(all_device_ids)

    # Validate and iterate to confirm the combined definition works end-to-end
    combined.validate(sdk)
    assert combined.is_validated

    window_count = 0
    for window in sdk.get_iterator(combined, window_duration=60, window_slide=60, time_units="s"):
        window_count += 1
    assert window_count > 0

    # Also test combining two already-validated definitions
    def_a.validate(sdk)
    def_b.validate(sdk)
    combined_validated = combine_definitions([def_a, def_b])
    assert combined_validated.is_validated


# ===================================================================
# SDK-dependent: cross_validate_dataset
# ===================================================================

def test_cross_validate_dataset():
    _test_for_both(DB_NAME, _test_cross_validate_dataset)


def _test_cross_validate_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk.get_all_devices().keys()}
    label_names = [info['name'] for info in sdk.get_all_label_names().values()]
    definition = DatasetDefinition(measures=measures, device_ids=device_ids, labels=label_names)

    n_folds = 3
    n_val_folds = 1
    n_test_folds = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        folds = cross_validate_dataset(
            definition,
            sdk,
            n_folds=n_folds,
            n_val_folds=n_val_folds,
            n_test_folds=n_test_folds,
            random_state=SEED,
            output_dir=tmpdir,
            filename_prefix="test_",
            verbose=False,
        )

        expected_combos = comb(n_folds, n_val_folds) * comb(n_folds - n_val_folds, n_test_folds)
        assert len(folds) == expected_combos

        # Verify each combination has the right structure
        for combo in folds:
            assert 'train' in combo and 'val' in combo and 'test' in combo
            assert 'fold_indices' in combo
            assert isinstance(combo['train'], DatasetDefinition)
            assert isinstance(combo['val'], DatasetDefinition)
            assert isinstance(combo['test'], DatasetDefinition)

            # All fold indices should cover the full set
            all_indices = (set(combo['fold_indices']['train']) |
                           set(combo['fold_indices']['val']) |
                           set(combo['fold_indices']['test']))
            assert all_indices == set(range(n_folds))

        # Verify YAML files were written
        expected_files = expected_combos * 3  # train + val + test per combo
        yaml_files = [f for f in os.listdir(tmpdir) if f.endswith('.yaml')]
        assert len(yaml_files) == expected_files

        # Verify files can be loaded back
        for yaml_file in yaml_files:
            loaded = DatasetDefinition(filename=os.path.join(tmpdir, yaml_file))
            assert len(loaded.data_dict['measures']) > 0


def test_cross_validate_dataset_invalid_folds():
    """Invalid fold configurations raise ValueError."""
    _test_for_both(DB_NAME, _test_cross_validate_dataset_invalid_folds)


def _test_cross_validate_dataset_invalid_folds(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    # Not enough folds for the requested assignment
    with pytest.raises(ValueError, match="too small"):
        cross_validate_dataset(definition, sdk, n_folds=2, n_val_folds=1, n_test_folds=1)

    # Both val and test set to 0
    with pytest.raises(ValueError, match="At least one"):
        cross_validate_dataset(definition, sdk, n_folds=5, n_val_folds=0, n_test_folds=0)


def test_cross_validate_dataset_zero_val_folds():
    """n_val_folds=0 produces combinations with val=None."""
    _test_for_both(DB_NAME, _test_cross_validate_dataset_zero_val_folds)


def _test_cross_validate_dataset_zero_val_folds(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    folds = cross_validate_dataset(
        definition, sdk, n_folds=3, n_val_folds=0, n_test_folds=1, random_state=SEED, verbose=False)

    assert len(folds) == 3  # C(3,0)*C(3,1) = 3
    for combo in folds:
        assert combo['val'] is None
        assert combo['fold_indices']['val'] == []
        assert isinstance(combo['train'], DatasetDefinition)
        assert isinstance(combo['test'], DatasetDefinition)


def test_cross_validate_dataset_zero_test_folds():
    """n_test_folds=0 produces combinations with test=None."""
    _test_for_both(DB_NAME, _test_cross_validate_dataset_zero_test_folds)


def _test_cross_validate_dataset_zero_test_folds(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    measures = [measure_info['tag'] for measure_info in sdk.get_all_measures().values()]
    device_ids = {np.int64(device_id): "all" for device_id in sdk.get_all_devices().keys()}
    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    folds = cross_validate_dataset(
        definition, sdk, n_folds=3, n_val_folds=1, n_test_folds=0, random_state=SEED, verbose=False)

    assert len(folds) == 3  # C(3,1)*C(2,0) = 3
    for combo in folds:
        assert combo['test'] is None
        assert combo['fold_indices']['test'] == []
        assert isinstance(combo['train'], DatasetDefinition)
        assert isinstance(combo['val'], DatasetDefinition)