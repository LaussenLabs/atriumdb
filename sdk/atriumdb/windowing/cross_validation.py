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
from itertools import permutations
from typing import List, Optional

from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.windowing.definition_splitter import partition_dataset
from atriumdb.windowing.definition_combine import combine_definitions


def cross_validate_dataset(
    definition,
    sdk,
    n_folds=5,
    n_val_folds=1,
    n_test_folds=1,
    output_dir=None,
    filename_prefix="",
    priority_stratification_labels=None,
    additional_labels=None,
    random_state=None,
    n_trials=None,
    num_show_best_trials=None,
    gap_tolerance=60_000_000_000,
    verbose=False,
):
    """
    Generate cross-validation folds from a validated :class:`DatasetDefinition`.

    This function partitions a dataset into ``n_folds`` equal-sized folds (by patient),
    then enumerates every unique assignment of folds to training, validation, and test
    roles. For each assignment the constituent folds are recombined into single
    :class:`DatasetDefinition` objects for training, validation, and testing.

    The fold partitioning is performed once via :func:`partition_dataset` with equal
    ratios, so the same folds are reused across all combinations. Which folds serve
    as training, validation, or test changes from one combination to the next.

    :param definition: A :class:`DatasetDefinition` (validated or unvalidated; will be
        validated automatically if needed).
    :param sdk: An :class:`AtriumSDK` instance.
    :param int n_folds: Number of folds to partition into. Must be at least 2. Default is 5.
    :param int n_val_folds: Number of folds assigned to validation in each combination.
        Set to 0 to omit validation sets entirely. Default is 1.
    :param int n_test_folds: Number of folds assigned to testing in each combination.
        Set to 0 to omit test sets entirely. Default is 1.
    :param str output_dir: If provided, each combination's train/val/test definitions
        are saved as YAML files in this directory.
    :param str filename_prefix: Optional string prepended to each output filename
        (e.g. a date string ``"2025-06-01_"``).
    :param priority_stratification_labels: Passed through to :func:`partition_dataset`
        for stratified fold creation.
    :param additional_labels: Passed through to :func:`partition_dataset`.
    :param random_state: Integer seed for reproducible fold creation.
    :param n_trials: Number of trials for :func:`partition_dataset` fold creation.
    :param num_show_best_trials: Number of best trials to display when verbose.
    :param gap_tolerance: Gap tolerance in nanoseconds for :func:`partition_dataset`.
    :param bool verbose: If True, print fold and combination details.
    :returns: A list of dicts, one per combination. Each dict has the keys
        ``"train"``, ``"val"``, ``"test"`` mapping to :class:`DatasetDefinition`
        objects (or ``None`` if that role has zero folds assigned), plus
        ``"fold_indices"`` mapping to a dict with keys ``"train"``, ``"val"``,
        ``"test"`` indicating which fold indices were used.
    :rtype: list[dict]
    :raises ValueError: If ``n_folds`` is too small for the requested fold assignment.

    **Examples**:

    Basic 5-fold cross-validation:

    >>> from atriumdb import DatasetDefinition, AtriumSDK, cross_validate_dataset
    >>> definition = DatasetDefinition(measures=["ECG"], patient_ids={1: "all", 2: "all"})
    >>> sdk = AtriumSDK(dataset_location="/path/to/dataset")
    >>> folds = cross_validate_dataset(
    ...     definition, sdk,
    ...     n_folds=5,
    ...     n_val_folds=1,
    ...     n_test_folds=1,
    ...     random_state=42,
    ...     output_dir="./cv_output",
    ...     filename_prefix="2025-06-01_",
    ... )
    >>> len(folds)  # 5 * 4 = 20 unique combinations
    20
    >>> folds[0].keys()
    dict_keys(['train', 'val', 'test', 'fold_indices'])

    Accessing the definitions for the first combination:

    >>> train_def = folds[0]["train"]
    >>> val_def = folds[0]["val"]
    >>> test_def = folds[0]["test"]
    """
    n_train_folds = n_folds - n_val_folds - n_test_folds
    if n_train_folds < 1:
        raise ValueError(
            f"n_folds={n_folds} is too small for n_val_folds={n_val_folds} and "
            f"n_test_folds={n_test_folds}. At least one fold must remain for training."
        )
    if n_val_folds < 0:
        raise ValueError("n_val_folds must be >= 0.")
    if n_test_folds < 0:
        raise ValueError("n_test_folds must be >= 0.")
    if n_val_folds == 0 and n_test_folds == 0:
        raise ValueError("At least one of n_val_folds or n_test_folds must be > 0.")

    # Step 1: Partition into n_folds equal folds.
    equal_ratios = [1] * n_folds

    partition_result = partition_dataset(
        definition,
        sdk,
        partition_ratios=equal_ratios,
        priority_stratification_labels=priority_stratification_labels,
        additional_labels=additional_labels,
        random_state=random_state,
        verbose=verbose,
        n_trials=n_trials,
        num_show_best_trials=num_show_best_trials,
        gap_tolerance=gap_tolerance,
    )

    # partition_dataset returns (defs,) or ((defs,), info) depending on verbose + n_trials.
    if verbose and n_trials is None:
        fold_definitions, fold_info = partition_result
    else:
        fold_definitions = partition_result
        if isinstance(fold_definitions, tuple) and len(fold_definitions) == 2 and isinstance(fold_definitions[1], list):
            fold_definitions, fold_info = fold_definitions
        else:
            fold_info = None

    # Ensure fold_definitions is a list/tuple of DatasetDefinition objects.
    if not isinstance(fold_definitions, (list, tuple)):
        fold_definitions = list(fold_definitions)

    if verbose:
        print(f"\nGenerated {n_folds} folds.")
        for i, fold_def in enumerate(fold_definitions):
            source_count = sum(
                len(sources) for sources in fold_def.data_dict.get('patient_ids', {}).items()
            ) + sum(
                len(sources) for sources in fold_def.data_dict.get('device_ids', {}).items()
            )
            print(f"  Fold {i}: {source_count} source entries")

    # Step 2: Generate all unique combinations of fold assignments.
    combinations = _generate_fold_combinations(
        n_folds, n_val_folds, n_test_folds
    )

    if verbose:
        print(f"\nGenerated {len(combinations)} unique fold combinations "
              f"(val={n_val_folds}, test={n_test_folds}, train={n_train_folds}).")

    # Step 3: For each combination, recombine folds into train/val/test definitions.
    results = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for combo_idx, combo in enumerate(combinations):
        val_indices = combo['val']
        test_indices = combo['test']
        train_indices = combo['train']

        train_def = _combine_fold_definitions([fold_definitions[i] for i in train_indices])
        val_def = _combine_fold_definitions([fold_definitions[i] for i in val_indices]) if val_indices else None
        test_def = _combine_fold_definitions([fold_definitions[i] for i in test_indices]) if test_indices else None

        result = {
            'train': train_def,
            'val': val_def,
            'test': test_def,
            'fold_indices': {
                'train': list(train_indices),
                'val': list(val_indices),
                'test': list(test_indices),
            },
        }
        results.append(result)

        # Step 4: Save to output directory if requested.
        if output_dir is not None:
            combo_num = combo_idx + 1
            for role, role_def in [('train', train_def), ('val', val_def), ('test', test_def)]:
                if role_def is None:
                    continue
                filename = f"{filename_prefix}dataset_def_{role}_fold_{combo_num}.yaml"
                filepath = os.path.join(output_dir, filename)
                role_def.save(filepath, force=True)

            if verbose:
                print(f"  Combination {combo_num}: train={list(train_indices)}, "
                      f"val={list(val_indices)}, test={list(test_indices)} -> saved to {output_dir}")

    if verbose:
        print(f"\nCross-validation complete: {len(results)} combinations generated.")

    return results


def _generate_fold_combinations(n_folds, n_val_folds, n_test_folds):
    """
    Generate all unique ways to assign ``n_folds`` folds into validation, test,
    and training roles.

    Each combination assigns ``n_val_folds`` folds to validation, ``n_test_folds``
    to testing, and the remaining folds to training. The order of folds within a
    role does not matter (i.e. val=[0,1] is the same as val=[1,0]), but the
    assignment of val vs test is distinct (val=[0] test=[1] differs from
    val=[1] test=[0]).

    :returns: A list of dicts with keys ``"val"``, ``"test"``, ``"train"``,
        each mapping to a tuple of fold indices.
    :rtype: list[dict]
    """
    from itertools import combinations as iter_combinations

    fold_indices = list(range(n_folds))
    result = []
    seen = set()

    # Choose which folds go to val.
    for val_combo in iter_combinations(fold_indices, n_val_folds):
        remaining_after_val = [i for i in fold_indices if i not in val_combo]
        # Choose which of the remaining folds go to test.
        for test_combo in iter_combinations(remaining_after_val, n_test_folds):
            train_combo = tuple(i for i in fold_indices if i not in val_combo and i not in test_combo)

            # Create a canonical key to avoid duplicates.
            key = (tuple(sorted(val_combo)), tuple(sorted(test_combo)), tuple(sorted(train_combo)))
            if key not in seen:
                seen.add(key)
                result.append({
                    'val': tuple(sorted(val_combo)),
                    'test': tuple(sorted(test_combo)),
                    'train': tuple(sorted(train_combo)),
                })

    return result


def _combine_fold_definitions(fold_defs):
    """
    Combine one or more fold :class:`DatasetDefinition` objects into a single definition.

    If only one definition is provided, a deep copy is returned.
    """
    import copy

    if len(fold_defs) == 1:
        result = DatasetDefinition(
            measures=copy.deepcopy(fold_defs[0].data_dict['measures']),
            labels=copy.deepcopy(fold_defs[0].data_dict['labels']),
            patient_ids=copy.deepcopy(fold_defs[0].data_dict['patient_ids']) or None,
            mrns=copy.deepcopy(fold_defs[0].data_dict['mrns']) or None,
            device_ids=copy.deepcopy(fold_defs[0].data_dict['device_ids']) or None,
            device_tags=copy.deepcopy(fold_defs[0].data_dict['device_tags']) or None,
        )
        if fold_defs[0].is_validated:
            result.validated_data_dict = copy.deepcopy(fold_defs[0].validated_data_dict)
            result.is_validated = True
            result.validated_gap_tolerance = fold_defs[0].validated_gap_tolerance
        return result

    return combine_definitions(fold_defs)