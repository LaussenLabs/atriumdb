# Changelog for AtriumDB

All notable changes to this project will be documented in this file.

## [Unreleased] — aperiodic and text support

### Added

- **String (text) measures.** A measure can now hold `str` values. Write them with the ordinary
  write methods (`AtriumSDK.write_time_value_pairs`, `AtriumSDK.write_data`) by passing a
  `list[str]` or a string/object numpy array; each distinct value is assigned an `int64`
  dictionary code. Read them back with the dedicated
  `AtriumSDK.get_string_data`, which returns `(times, values)` with `values` a 1D object array
  of `str`. Per-measure dictionaries are append-only JSON Lines files under
  `<dataset_location>/meta/string_dict/measure_<measure_id>.jsonl`.

- **Measure metadata: `signal_kind` and `value_type`.** Every measure now carries two
  independent axes — the temporal shape (`waveform` / `sample` / `event` / `state`) and the value
  encoding (`numeric` / `string`). Both are accepted by `AtriumSDK.insert_measure`, returned by
  `get_measure_info` / `get_all_measures`, and readable as a pair via the new
  `AtriumSDK.get_measure_kind`. `AtriumSDK.set_measure_kind` corrects them after the fact
  (`signal_kind` is always safe to change; `value_type` cannot be changed once data exists).
  `get_all_measures` records also now expose `period_ns`.

- **Aperiodic and string measure windowing.** `AtriumSDK.get_iterator` rasterizes aperiodic and
  string measures onto the window grid so every measure still yields a fixed-length `values`
  array. Per-`signal_kind` fill rules (`carry_forward`, `sparse`, `aggregate:last|mean|min|max`,
  `presence`, `count`) with a 1-second nominal raster period, configurable through the new
  `aperiodic_fill`, `fill_overrides` and `period_overrides` parameters. Unknown / censored cells
  are marked with a sentinel (`NaN` for numeric, `-1` → `"<unknown>"` for string codes).

- **String decoding accessors.** `Window.decode_string_signal(sdk, measure_key)` and
  `DatasetIterator.decode_window_strings(window, measure_key)` decode a window's `int64`
  dictionary codes. `sample` / `state` string measures carry codes; `event` string measures are
  rasterized to numeric occupancy and are deliberately not decodable.

- **Event queries.** `AtriumSDK.get_measure_string_vocabulary` (every value ever written, read
  from the dictionary with no data scan), `AtriumSDK.get_string_values_present` (distinct values
  observed for a source over a range), and `AtriumSDK.get_event_intervals`, which pairs a
  `from_value` with the next `to_value` using a collapse rule and returns
  `{start_time_n, end_time_n, start_censored, end_censored}` spans. Containment follows a
  `device_patient` → `encounter` → whole-stream cascade, selectable with `within`.

- **Event-anchored dataset definitions.** A source's time-spec list may contain event-anchored
  regions: `anchor` (a `[t - pre, t + post]` window around every occurrence of an event value) and
  `from`/`to` (the span between an opening and a closing event), with `within`, `pre`, `post`,
  `max_duration` and `on_censored` options. Available in both the Python and YAML forms.

- **Encounters.** `AtriumSDK.insert_encounter`, `get_encounters` and
  `get_device_patient_encounters`, plus `within: encounter` scoping for event queries and
  event-anchored regions.

- **Transfer of aperiodic, string and encounter data.** `transfer_data` now carries string
  measures (metadata, dictionaries and codes, with vocabulary union and code remapping into the
  destination) and the encounter family (`encounter`, `device_encounter`, `bed`, `unit`,
  `institution`) by default. New parameters: `include_encounters`, `keep_identified` (per-table
  allowlist for encounter-family de-identification), and `string_value_policy`
  (`"transfer"` / `"redact"` / `"skip"` / a callable) governing what happens to string measure
  values. **`string_value_policy` defaults to `"transfer"` in every mode**, `deidentify=True`
  included — de-identification is scoped to PHI and does not alter signal content. The other
  policies are an explicit opt-in for a caller with their own reason to scrub one free-text
  measure.

- **De-identification scope.** `deidentify` covers patient-level metadata, the patient ID remap,
  `encounter.visit_number` (scrambled to a random int via a consistent per-transfer map) and the
  uniform `time_shift`. `log_hl7_adt` is never transferred. Everything else — string measure
  values, label names and label text, device tags, and bed / unit / institution names — transfers
  **verbatim**.

- **Read-side duplicate handling.** `get_data` and `get_string_data` take `allow_duplicates`
  (default `True`) and `duplicate_keep`. With `allow_duplicates=False`, samples sharing a
  timestamp collapse to one, and the survivor follows the dataset's `overwrite` policy — the
  **newest** stored copy by default (`"overwrite"` / `"ignore"`), the earliest under `"protect"`.
  `duplicate_keep="last"` / `"first"` overrides that per call. String measures export to `csv`,
  `npz` and `parquet` as decoded strings; `wfdb` cannot hold text and warns about each measure it
  omits, striking it from the exported manifest.

- **Cross-process write locking.** The small-write block merge and the string dictionary append
  are serialized with advisory file locks under `meta/locks/` and
  `meta/string_dict/*.jsonl.lock`, making multi-process ingest against one dataset safe on a
  filesystem with working `flock`.

- **Documentation:** a new Operations page (backup surface, durability, concurrency, replay /
  duplicate-timestamp semantics), and new sections on declaring aperiodic measures, choosing a
  `signal_kind`, reading string windows, and the inventory of text surfaces in a transfer.

### Changed

- **`get_data(..., allow_duplicates=False)` now keeps the NEWEST copy of a duplicated
  timestamp, not the first stored one.** Survivor semantics follow the dataset's `overwrite`
  policy so a read agrees with what a write would have done had the two copies met in one block.
  Pass `duplicate_keep="first"` to restore the previous behaviour.
- `insert_measure` rejects `freq=0` with a message explaining how to declare an aperiodic
  measure instead.
- Writing numeric values to an established string measure (or vice versa) raises `ValueError`.
- `get_data` on a string measure with `analog=True` (the default) or `return_nan_filled` raises
  and points at `get_string_data`.
- Aperiodic measures use a widened interval-index gap tolerance, so their interval array is a
  coarse presence map rather than a per-sample one.

### Known issues

- **CSV export loses nanosecond timestamp precision.** The CSV writer converts the `int64`
  nanosecond times to `float64` before writing; a modern nanosecond epoch exceeds the range a
  `float64` represents exactly, so the low digits are rounded away. Sample values are unaffected.
  Use `tsc`, `npz` or `parquet` when exact timestamps matter.

## [2.1.0] - 2023-10-20

### Added

- **DatasetDefinition class**: Introduced a new `DatasetDefinition` class to define data sources, signals, and times. This encapsulates information vital for features like data exports and data iterators.
  
- **Definition File Format**: Introduced a file format to work alongside the `DatasetDefinition`. This allows the raw data of a `DatasetDefinition` object to be saved to or loaded from a file with `.yaml` or `.yml` extension.

- **DatasetIterator class**: Added a new `DatasetIterator` class which implements the `__next__`, `__len__`, and `__getitem__` methods. This can be used to iterate through time windows of data as described by the `DatasetDefinition` class.

- **AtriumSDK.get_iterator Method**: A new method in the main module's class. The `AtriumSDK.get_iterator` method consumes a `DatasetDefinition`, validates the definition with the actual data available in the datastore targeted by the `AtriumSDK` object, and returns a `DatasetIterator` object if the `DatasetDefinition` was successfully validated.

- **Patient Parameters**: Now supports `height` and `weight` as patient parameters/keys in dictionaries returned or set by AtriumSDK methods like `insert_patient`, `get_patient_info`, and `get_all_patients`.

### Fixed

- Resolved a bug encountered when the `atriumdb` library accessed datastores set up by different software versions. This was addressed by substituting `SELECT *` wildcards in SQL queries with the precise column names required by each query.
