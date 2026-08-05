# Design: Aperiodic signals, events, and text support in AtriumDB

Status: **Draft / research** — branch `feature/aperiodic-and-text-support` (based on
`feature/smart-write-defaults`). This document decides *how* aperiodic data fits the
existing framework and *how* text data is stored, compressed, and queried. It is a
design deliverable; no runtime behaviour is changed yet.

---

## 1. The two problems, restated

1. **Aperiodic signals / events break the interval index.** Alarms, non-invasive
   blood pressure (NIBP), manual entries, and "event A on / off" records are not
   sampled on a fixed period. The interval index was designed to answer "where does
   continuous data exist" for waveforms; for genuinely aperiodic data that question
   has no natural answer, and a naive index either explodes (one row per sample) or
   is meaningless (one huge row).

2. **We have no text support.** We need dynamically-sized strings (fixed-length only
   as a last resort, with a large cap and still efficient), plus the ability to:
   - query the span between `"Event A on"` and `"Event B off"`;
   - enumerate the unique event types;
   - filter events by type and by data tags (patient, device, measure, source);
   - handle **left-censored** state: recording starts with "Event A" *already on*, so
     the opening transition was never observed.

A recurring constraint from the requester: **write-once / query-many.** These signals
are appended, rarely mutated or deleted, so we should not lean on the transactional
SQL tables (row-level overhead, index maintenance, lock contention) to hold the bulk
of event/text payloads. The columnar `.tsc` block store is the right home for the
data; SQL should hold only coarse routing metadata.

---

## 2. How the interval index is actually used (confirmed)

The requester's recollection was essentially correct. The interval index
(`interval_index` table, populated in `write_data` via `get_block_and_interval_data`)
has exactly these consumers:

1. **`AtriumSDK.get_interval_array`** ([atrium_sdk.py:5359](../../sdk/atriumdb/atrium_sdk.py))
   — user-facing "where is data available." Reads `interval_index` rows, clips to
   `[start, end]`, and merges rows closer than `gap_tolerance_nano`.

2. **Dataset Definition validation** — `verify_definition._get_validated_entries`
   ([windowing/verify_definition.py:203](../../sdk/atriumdb/windowing/verify_definition.py))
   calls `get_interval_array` per measure and unions the results
   (`intervals_union_list`) to produce the concrete time ranges a source actually has
   data for. `"all"`, `{start,end}`, and `{time0,pre,post}` region specs are all
   intersected against this union.

3. **The validator / source mapper** — `map_validated_sources`
   ([windowing/map_definition_sources.py:26](../../sdk/atriumdb/windowing/map_definition_sources.py))
   turns those ranges into `(device_id, patient_id)` tuples with concrete sub-ranges.

4. **The iterator** — `DatasetIterator._extract_cache_info`
   ([windowing/dataset_iterator.py:190](../../sdk/atriumdb/windowing/dataset_iterator.py))
   consumes the validated/mapped time ranges to lay out window batches. It never
   touches `interval_index` directly; it trusts the ranges the definition produced.

5. Export/transfer (`transfer/adb/dataset.py`, `transfer/formats/dataset.py`) also
   calls `get_interval_array` to decide what to copy.

**Key consequence:** the interval index is a *planning* structure. Anything that wants
to drive the iterator (including event-anchored cohorts) only has to emit
`{source -> [(start, end), ...]}`. We do **not** need to overload `interval_index`
itself with event semantics; we need a second producer of time ranges that feeds the
same `verify_definition` → `map_validated_sources` → `DatasetIterator` pipeline.

---

## 3. What the `feature/smart-write-defaults` branch already solved

This branch is a good base because it already did the numeric half of the aperiodic
problem:

- **Waveform vs aperiodic classification** from observed median sample spacing, with
  the cutoff `APERIODIC_MIN_PERIOD_NS = 200_000_000` (5 Hz)
  ([adb_functions.py:400](../../sdk/atriumdb/adb_functions.py)).
- **Aperiodic-aware time encoding**: `choose_time_encoding` picks a raw gap array for
  regular data, a zstd gap array for structured jitter, and a **zstd timestamp array**
  for genuinely aperiodic data — so aperiodic numeric signals already store compactly.
- **Interval-index gap tolerance** is widened for aperiodic data
  (`choose_interval_gap_tolerance` + `widen_gap_tolerance_for_observed_spacing`) so
  irregular arrivals don't flood the index, plus a density warning when a write still
  produces too many interval rows.

So aperiodic **numeric** data already "fits the framework": it is a signal with an
estimated period, stored as a compressed timestamp array, with a coarse interval
index. What is missing is (a) **non-numeric** (text) values, (b) **event/state**
semantics including left-censoring, and (c) **event-anchored dataset construction**.

---

## 4. Conceptual model: separate *shape* from *value encoding*

The cleanest way to make all of this fit is to stop conflating two independent axes.

**Axis 1 — temporal shape** (how the timeline is described):
- **Waveform / periodic** — fixed period, gap array. (Today's default.)
- **Sample series / aperiodic point data** — irregular timestamps, one value each
  (NIBP readings, lab values, point events). Timestamp array + zstd. (Largely done.)
- **State / interval series** — a value that holds between two times (a "mode",
  "alarm active", "Event A on"). A **step function**: store transitions, each value
  is valid until the next transition. This is where left-censoring lives.

**Axis 2 — value encoding** (what each value *is*):
- `int64`, `double` (today).
- **`string`** (new) — variable-length UTF-8, dictionary + zstd.

These compose. "Event A on/off" is a *state series* whose *values* are *strings* (or a
dictionary enum). NIBP is a *sample series* of *doubles*. A device operating-mode log
is a *state series* of *strings*. Free-text annotations at high volume are a *sample
series* of *strings*. Modeling text as a value type (Axis 2) rather than a new
subsystem means it rides the **entire existing block pipeline**: encode/decode,
`block_index`, `interval_index`, block merging, the iterator, transfer/export.

This gives us the write-once/query-many property for free: event/text payloads live in
immutable `.tsc` blocks, and SQL only gains a tiny dictionary table plus the block/
interval rows it already keeps.

---

## 5. Track A — text as a first-class value type in the block format

### 5.1 New value type codes

In `block_wrapper.py` and `tsc-lib` (`block_header.h`, value codecs):

```
V_TYPE_STRING        = 5   # raw: variable-length UTF-8, presented as an array of Python str / object dtype
V_TYPE_STRING_DICT   = 6   # encoded: dictionary-of-uniques + int codes (+ zstd on both streams)
```

`V_TYPE_STRING` is the *raw* (logical) type; `V_TYPE_STRING_DICT` is an *encoded*
representation, exactly mirroring how `V_TYPE_INT64` (raw) maps to
`V_TYPE_DELTA_INT64` (encoded) today. A caller writes string values; the encoder
chooses dictionary vs. plain based on cardinality.

### 5.2 On-disk block layout for a string value block

A string block's value payload is a small self-describing structure so decode needs no
external state:

```
[ n_codes: u64 ]                         # number of values (== num_vals)
[ n_unique: u64 ]                        # dictionary size
[ code_width: u8 ]                       # 1/2/4 bytes per code, by n_unique
[ dict_offsets: (n_unique+1) x u32 ]     # prefix offsets into dict_blob
[ dict_blob: bytes ]                     # concatenated UTF-8 of unique strings, zstd
[ codes: n_codes x code_width ]          # dictionary indices, zstd
```

Rationale, tied to the research:
- **Dictionary encoding** is the standard columnar approach for low-to-moderate
  cardinality strings (ClickHouse `LowCardinality(String)`, Parquet `PLAIN_DICTIONARY`,
  Arrow `DictionaryArray`). Clinical event/mode vocabularies are tiny (tens to a few
  thousand distinct strings) with massive repetition, which is the ideal case:
  "N strings + M small integer codes" instead of M full strings.
- **Offsets + blob** gives true variable length (no fixed cap) while staying a flat,
  vectorizable structure — no per-value length prefixes scattered through the data.
- **zstd on both streams** matches the branch's existing time-compression choice and
  captures residual redundancy in both the dictionary and the code stream.
- Cardinality guard: research puts dictionary encoding's crossover around tens of
  thousands of distinct values per block. Above a configurable ratio
  (`n_unique / n_codes`) we fall back to `V_TYPE_STRING` = length-prefixed UTF-8 blob
  + zstd (no dictionary), so genuinely high-entropy free text still works, just without
  the dictionary win. This decision is made per block in the same "measure the encoded
  size and keep the smaller" spirit as `choose_time_encoding`.

### 5.3 Header changes

`BlockMetadata` already carries `v_raw_type`, `v_encoded_type`, `v_raw_size`,
`v_encoded_size`, `bytes_per_value`. For strings:
- `bytes_per_value` becomes `code_width` for the dict form (0 or a sentinel for the
  plain blob form).
- `max`/`min`/`mean`/`std`/`scale_m`/`scale_b` are numeric-only; for strings they are
  written as sentinels and ignored. No struct layout change is required — this keeps
  old readers' parsing valid and the change is purely additive (new enum values).

### 5.4 C library

Add `Value_String` codec module alongside `Value_Int64` / `Value_Double`:
- `value_string_get_size`, `value_string_encode`, `value_string_decode`, buffer sizing.
- Wire new `case V_TYPE_STRING` / `V_TYPE_STRING_DICT` branches into
  `value_encode.c`, `value_decode.c`, `value_buf.c`.
- `convert_value_data_to_analog` and `fill_nan_array_with_analog` are numeric-only and
  must reject / bypass string blocks (return the codes or raise) — the Python layer
  guards `analog=` for string measures.

Because the C ABI (`BlockMetadata`) is unchanged, the macOS/Linux/Windows dylibs just
gain new switch cases; no struct-packing churn.

### 5.5 Python layer

- `block.py` `encode_blocks` / `decode_blocks`: accept/return an `object`/`str` numpy
  array (or Arrow-style (offsets, blob) pair) for string measures; skip analog scaling.
- `_resolve_value_types` learns `str`/`object` dtype → `V_TYPE_STRING` /
  `V_TYPE_STRING_DICT`.
- `get_data` returns a string array for string measures; `return_nan_filled` /
  `analog` are disabled (raise a clear error) for them.
- Block **merge** (this branch's small-write merge) works unchanged in principle: two
  string blocks concatenate their value arrays and re-encode, re-deriving the
  dictionary from the merged data (the branch already re-chooses encodings after merge).

### 5.6 Why not a SQL text column or a blob table?

- A SQL `TEXT` column per event pays row + index + transaction overhead per event; at
  clinical scale (millions of alarm toggles) this is exactly the "transactional
  overhead we don't need." The columnar block store amortizes to a few bytes per event
  after dictionary + zstd.
- It also keeps text **on the same read path** as signals, so the iterator/windowing/
  transfer machinery needs no parallel plumbing.

---

## 6. Track B — event & state semantics (incl. left-censoring)

Text-as-a-value-type gives storage; we still need *meaning* on top. Two record shapes:

### 6.1 Point events — a string sample series

`insert_events(measure_or_stream, device/patient, times_ns, values: list[str])` writes
a `V_TYPE_STRING` sample series. `"Event A on"`, `"Event A off"`, `"NIBP taken"` are
just string values at timestamps. Enumerating unique event types is reading the block
dictionaries (see §7). This is the general case and needs nothing beyond Track A.

### 6.2 State intervals & the existing `label` system

The repo already has an interval-shaped, hierarchical, tag-associated text system: the
**label** tables (`label_set`, `label_source`, `label`) and API
(`insert_label(s)`, `get_labels`, `get_label_time_series`, `get_all_label_names`,
hierarchical `label_name` parent/child). A label is `(name, device/patient, measure?,
source?, start_time_n, end_time_n)` — i.e. **a named state interval with tags**. It is
already wired into `DatasetDefinition` (`labels:`) and the iterator
(`get_label_time_series`).

This is the natural home for **derived / analyst-facing state intervals** ("Event A was
on from t0 to t1"), and much of the requested query surface already exists:
- *Unique event types* → `get_all_label_names` (already hierarchical).
- *Filter by type + tags* → `get_labels(name_list=, device_list=/patient_id_list=,
  measure_list=, label_source_list=, start_time=, end_time=)`.

Two gaps to close:

**(a) Provenance / scale.** Labels live entirely in SQL. That is fine for *derived*,
lower-volume annotations (the write-once-query-many argument is weaker when a human or
one algorithm produced them). For *raw, high-volume* device event streams, ingest them
as a Track A string sample series (block store), and provide a **materialization step**
that folds a raw on/off event series into label state intervals (or computes intervals
on the fly at query time — see §6.3). Recommendation: raw events → block store; curated
state intervals → labels. Document the boundary clearly.

**(b) Left-censoring.** Neither labels nor a raw event series currently expresses
"already on when recording began." Design below.

### 6.3 Deriving state intervals from on/off events, with left-censoring

Given a string state series with transition values (`"A on"`, `"A off"`), the interval
for state "A active" is `[on_time, next off_time)`. Two boundary problems:

- **Left-censored (open start):** the first observed record for A is `"off"`, or A is
  active at the very first sample with no preceding `"on"`. The true onset precedes our
  data. Represent this with an explicit **unknown boundary sentinel** rather than
  guessing: an interval `(start=UNKNOWN, end=first_off)` where `UNKNOWN` is encoded as
  a distinguished value (e.g. `start_time_n = LEFT_CENSORED` sentinel, or a boolean
  `start_censored` column on the derived interval). Downstream code and the interval
  math treat a censored start as "extends to −∞ but clipped to the query/recording
  window," so a query for `[q0, q1]` yields `[q0, first_off)` and is flagged censored.
- **Right-censored (open end):** A turns on and never turns off within the data — end is
  unknown/ongoing. Symmetric sentinel; clip to recording end or query end, flag
  censored.

This mirrors the survival-analysis framing from the research (left/right/interval
censoring): we *record the censoring* instead of fabricating a boundary. The WFDB
annotation model (onset + optional duration, aux string payload) is the precedent for
"aperiodic marker with a text payload aligned to a sample clock"; we generalize it with
explicit censor flags.

Concretely, a small helper `derive_state_intervals(event_series, recording_bounds)`
returns `[(state_value, start_ns, end_ns, start_censored, end_censored), ...]`,
computed from the raw block-store event series and the device/patient recording window
(available from the interval index / `block_index` extents). This is pure computation
over already-stored data — no new persisted structure required unless the analyst wants
to snapshot the result as labels.

---

## 7. Track C — querying events, and event-anchored datasets

### 7.1 Enumerate unique event types cheaply

The per-block string dictionaries are a natural pre-aggregated index. Two options:

- **Cheap, no schema change:** scan the dictionaries of the relevant blocks (dictionary
  blobs are tiny) and union them. Fast because we skip the code streams entirely.
- **Optional acceleration:** a small `string_value_dict` SQL table
  `(measure_id, value_text, first_seen_n, last_seen_n)` maintained on write — a compact,
  low-cardinality table (one row per distinct event type per stream, not per event), so
  it stays small and write-once-friendly. This makes "list all event types for device X
  in window W" a single indexed query. This is the *only* new SQL table Track A/B need,
  and it is bounded by vocabulary size, not event count.

### 7.2 "Give me intervals from Event A to the next Event B"

This is a scan over the string sample series for one source: read the (dictionary-coded)
event series for `[q0, q1]`, then a vectorized pass pairs each `A` with the next `B`
(optionally constrained to the same encounter — see below). Because values are integer
dictionary codes after decode, the pairing is `np.where(codes == code_A)` /
`searchsorted` against `codes == code_B`. Output is a list of `(start, end)` intervals —
the exact shape the definition pipeline consumes.

"within a single encounter" is enforced by intersecting candidate `(A, B)` spans with
the `encounter` / `device_patient` intervals already in SQL, so a `B` from a later
admission does not close an `A` from an earlier one.

### 7.3 Event-anchored `DatasetDefinition` (the headline feature)

Today a region spec supports `all`, `{start,end}`, and `{time0,pre,post}`
(a *fixed* timestamp; see `verify_definition.py:246`). We extend the vocabulary with
**event anchors** that are resolved at validation time into concrete `(start, end)`
ranges — feeding the *same* `verify_definition` → `map_validated_sources` →
`DatasetIterator` path, so **the iterator and validator need no changes**.

New region specs (resolved in `_get_validated_entries` / a new resolver):

```yaml
# X minutes pre/post every occurrence of an event
- anchor: "Event A on"
  pre:  5m
  post: 5m

# between event A and the next instance of event B, within one encounter
- from: "Event A on"
  to:   "Event B off"
  within: encounter          # or: none | device_patient | <label window>
  # optional: pre/post padding, max_duration, on_missing_to: {clip|drop}

# between an event and a fixed offset, or event-to-event with a cap
- from: "Alarm start"
  max_duration: 30m
```

Resolution algorithm per source:
1. Query the event series (block store) for the source over the global bounds.
2. Materialize anchor timestamps / `(from, to)` spans (§7.2), honoring `within`.
3. Expand by `pre`/`post`, apply `max_duration`, clip to the source's data union (the
   existing interval-index union) and global `[start_time_n, end_time_n]`.
4. Emit `[(start, end), ...]` exactly as the current region branches do.

Left-censoring surfaces here too: a `from: "Event A on"` where A is already active at
the window start yields a censored-start span; the definition can choose to
`drop`, `clip` to the recording/query start, or keep-and-flag. Default: clip + warn
(consistent with how `verify_definition` already warns and skips empties).

This is the mechanism behind "datasets based on textual events like X and Y minutes pre
and post some event, or between event A and the next instance of event B within a single
encounter."

---

## 8. What happens to the interval index for aperiodic data

- **Numeric aperiodic** signals keep using `interval_index` with the widened gap
  tolerance this branch already computes — `get_interval_array` stays meaningful as a
  *coarse presence* map ("NIBP readings exist roughly in these hours").
- **Event / string** series also populate `interval_index` (coarse presence) so
  `get_interval_array` answers "are there events here at all," but **event-level
  querying never goes through the interval index** — it reads the event blocks directly
  (§7). This resolves the "interval index loses its meaning for events" concern: the
  interval index stays a presence map, and precise event logic is a columnar scan.
- Consider a per-measure `signal_kind` on the `measure` table
  (`waveform | sample | state | event`) so `get_interval_array`, the density warning,
  and the definition resolver can pick sensible defaults without re-inferring from data
  every time. This is metadata only; the branch's data-driven inference remains the
  fallback when unset.

---

## 9. How other systems inform this (research summary)

- **WFDB / PhysioNet annotations** — the canonical clinical precedent: annotations are
  aperiodic markers (onset as sample number or time, optional duration, an `aux` free
  text/string payload) aligned to the signal's sample clock, covering exactly our event
  cases (alarms, arrhythmia, sleep stage, apnea, signal-quality). Validates modeling
  events as timestamped records with a text payload, stored *alongside* but *separate
  from* the sampled signal. ([wfdb.io](https://wfdb.io/),
  [PhysioBank Annotations](https://archive.physionet.org/physiobank/annotations.shtml))
- **InfluxDB / event-vs-metric TSDBs** — unify event and metric storage but keep the
  *representations* distinct; irregular events are queried directly and only optionally
  aggregated onto a regular grid. Matches our "coarse presence index + direct event
  scan" split. ([influxdata.com](https://www.influxdata.com/time-series-database/))
- **Columnar dictionary encoding** — ClickHouse `LowCardinality(String)`, Parquet
  `PLAIN_DICTIONARY`, Arrow `DictionaryArray`: store uniques once + integer codes;
  wins below ~tens of thousands distinct values per block, which is the clinical
  event/mode regime. Drives the §5.2 layout and the cardinality fallback.
  ([ClickHouse compression](https://clickhouse.com/resources/engineering/database-compression))
- **Survival analysis / censoring** — left/right/interval censoring is the principled
  framing for "state active before observation began": record that the boundary is
  unknown rather than inventing it. Drives §6.3.
  ([Columbia TTE](https://www.publichealth.columbia.edu/research/population-health-methods/time-event-data-analysis))

---

## 10. Proposed phasing

1. **String value type (Track A).** C codec + `block_wrapper`/`block.py`/`get_data` +
   `write_events`/string write path + round-trip tests. Unlocks storing text at scale.
   Self-contained; nothing else depends on it landing first except that everything else
   builds on it.
2. **Event query surface (Track C §7.1–7.2).** Unique-type enumeration (dictionary
   scan + optional `string_value_dict` table), and `from→to` / anchor interval
   derivation as standalone SDK methods.
3. **Left-censoring + state derivation (Track B §6.3).** `derive_state_intervals`,
   censor flags, integration with `get_labels` output shape.
4. **Event-anchored DatasetDefinition (Track C §7.3).** The user-facing payoff; reuses
   1–3 and the existing iterator.
5. **`signal_kind` metadata + `get_interval_array` polish (§8).** Optional, quality-of-
   life.

## 11. Open questions / decisions to confirm

- **Where is the raw-event / label boundary drawn?** Recommendation: raw high-volume
  device events → block store (Track A); curated/derived state intervals → `label`
  tables. Confirm this split matches how analysts expect to query.
- **String cardinality fallback threshold** — pick the `n_unique/n_codes` ratio and an
  absolute `n_unique` cap for switching dict → plain-blob; measure on real alarm/mode
  vocabularies (reuse the `experiments/` harness).
- **Fixed-length escape hatch** — the requester allowed fixed-length "if absolutely
  necessary." The dict layout makes it unnecessary in the common case; do we still want
  a `V_TYPE_STRING_FIXED(n)` for a hot path where callers guarantee a bound? Default:
  no, revisit only if profiling demands it.
- **`within` semantics** — should `within: encounter` use the `encounter` table, the
  `device_patient` mapping, or an arbitrary named label window? Design supports all
  three; pick the default.
- **API mode** — `get_labels` has an API path; the new event methods and
  event-anchored definitions need matching server endpoints before they work in `api`
  metadata mode. Out of scope for the first local-mode passes but note it.

---

## 13. Decisions log (confirmed with requester)

These refine/override the earlier sections where they conflict.

- **Default raster period = 1 second.** When a requested signal has no meaningful
  period (aperiodic sample/event/state kinds, or a measure with no declared period),
  the iterator rasterizes onto a **1 s** grid rather than failing or guessing. Explicit
  per-measure nominal periods still win when set.
- **Carry-forward is the default fill for `sample` kinds** (NIBP, labs) — each grid cell
  takes the most recent prior reading — **with an override exposed on `get_iterator`**
  (e.g. `aperiodic_fill="carry_forward" | "sparse" | "aggregate:last|mean|min|max"`).
- **Event pairs rasterize to a state-membership row.** For "Event A → Event B" (e.g.
  "Anesthesia START" → "Anesthesia STOP"), each cell encodes whether the sample is
  *inside* that state (1) or not (0). See the §14 audit for the representation and the
  censoring caveat. This requires (a) a way for the `DatasetDefinition` to request event
  pairs, and (b) transfer to move both underlying events — see §14/§15.
- **`within` scoping default = a graceful cascade, not a fixed source.** Prefer
  `device_patient` where it is populated (it can exist without encounters being kept
  current, which suits simpler datasets); use `encounter` when explicitly requested for
  admission-level semantics; and **fall back to the whole stream / the time ranges the
  `DatasetDefinition` already produced** when neither is available (device-only datasets
  with no patient mapping, or an unpopulated `device_patient` table). The code must run
  with an empty `device_patient` table. Note (dataset-specific): known orphaned
  `device_patient` rows / encounter-completeness gaps exist here, so the resolver
  **warns rather than silently drops** when scoping data is missing.

- **String dictionary = append-only, stable codes, in `meta/`.** The logical vocabulary
  is one entry per unique string, code = insertion order, appended in place; existing
  codes are immutable so historical blocks never need rewriting. Codes are plain-zstd'd
  in the block. **No recalculating / trained-zstd compression dictionary in v1** — it
  would force versioning every historical block to the dict version it was written with
  for little gain; revisit later as a versioned dict in `meta/` if ever needed. High
  cardinality free text bypasses the global dict and uses per-block plain UTF-8 + zstd.
  Vocabularies are **per-measure** (small, independently transferable, matches how event
  enumeration/queries are scoped) rather than one global dataset dictionary. Stored as an
  append-only file in `meta/` where record index is the code (length-prefixed / JSONL so
  embedded newlines are safe).
- **`log_hl7_adt` is never transferred.** Not even in identified transfers for now; a
  future opt-in flag may allow including it. Until then it is excluded unconditionally.
- **`write_data_easy` is being deprecated → legacy behavior only.** It intentionally does
  NOT accept string values (it derives a numeric value type from dtype). String writes go
  through `write_time_value_pairs` / `write_data`. Do not add string support to
  `write_data_easy` in any phase — this is a decision, not an omission.
- **Numeric/string on one measure must be rejected (found by the P1 audit).** A measure is
  either numeric or string; the P1 code silently accepts a mixed write and corrupts
  readability. Enforcement is a **P2 deliverable** via the `value_type` column (§19.3) —
  reject a write whose value-kind conflicts with the measure's established `value_type`.

## 14. Audit: is a 0/1 state row the best way to represent event-pair membership?

**Question posed:** for "in an Event A → Event B period," is a rasterized row of 1s/0s
the only/best representation? **Finding: 0/1 membership is the right *default*, and the
best implementation is to *reuse the existing label rasterizer* rather than build a new
one — but plain binary is insufficient at censored boundaries, which needs one addition.**

### 14.1 Why 0/1 is right and already mostly built

"Am I inside an interval at time t," sampled onto a grid, is *exactly* the shape
`get_label_time_series` already produces (`[0,1,1,0,...]`, consumed by the iterator's
label-threshold path). So event-pair membership is not a new rasterization problem; it
is the label-timeseries problem with the intervals derived from event pairing instead of
stored directly. **Route event-pair states through the label rasterizer.**

### 14.2 Where to store the source of truth — three options

1. **Materialize pairs → label intervals at derive time.** Pair A/B once, store the
   result as a label set; the existing label storage, rasterization, transfer, and
   `DatasetDefinition labels:` all work unchanged.
   - *Pros:* almost no new machinery; transfer is already solved (labels transfer).
   - *Cons:* pairing rule fixed at write time; needs a derive step; both events must be
     present to pair.
2. **Keep raw events (Track A string series), pair at query/definition time.**
   - *Pros:* pairing rule flexible per query; raw events retained; change pre/post/within
     without re-ingest.
   - *Cons:* new "pair → step → grid" code; transfer must move *both* raw event streams
     and the exported definition must record the pair request; censoring computed live
     from recording bounds.
3. **Hybrid (recommended).** Raw events are the source of truth (stored + transferred as
   ordinary measures), and an **event-pair request in the `DatasetDefinition`** derives
   intervals *during validation*, which are then fed to the existing label rasterizer.
   Optionally snapshot the derived intervals as a persistent label set for speed/repro.
   This gets raw-event flexibility **and** reuses the label rasterizer, at the cost of
   the definition/transfer wiring in §15.

### 14.3 The one place binary is not enough: censoring

A plain 0/1 row silently miscodes **left/right-censored** regions (state active before
recording began, or never closed) as "not in state." Recommendation: emit a
**parallel validity/known mask** (or a tri-state: `0` not-in-state, `1` in-state,
`sentinel` unknown), so a consumer can distinguish "known not in state" from "unknown."
This is the only representational addition beyond the label path.

### 14.4 Other encodings the same pairing unlocks (not defaults)

- **Multiple states → one channel per state** (mirrors multiple label sets), not a single
  multiclass column, unless the states are mutually exclusive.
- **Time-to-event ramps** ("ns since A" / "ns until B") as continuous features, for models
  that want proximity rather than membership. Offer as an opt-in fill, not the default.

### 14.5 Verdict

Confirm 0/1 membership as the default, implemented via the label rasterizer, with a
known/validity mask for censoring. Prefer the **hybrid** storage model. Alternatives
(materialize-to-labels; per-state channels; time-to-event ramps) are real and cheap to
offer later, but binary-membership-via-labels is the correct primary path.

## 15. Transfer & de-identification survey

Two gaps: transfer must (a) learn the new measure kinds, and (b) start moving tables it
currently skips (notably `encounter`). Below is the field-level survey of what needs
**de-identification** and **time-shifting**.

### 15.1 What transfer moves today (and how)

`transfer_data` ([transfer/adb/dataset.py](../../sdk/atriumdb/transfer/adb/dataset.py))
+ `transfer_patient_info` move: **measures, devices, patient (+patient_history),
device_patient, labels, blocks, intervals**. Time-shifting is applied *field-by-field,
per function* — there is no global pass — so every new time-bearing field must be wired
explicitly. De-id today: `patient_id` remapped; PHI patient columns dropped unless
whitelisted (`patient_info_to_transfer`); `dob/first_seen/last_updated`,
`patient_history.time`, `device_patient.start/end`, `label.start/end`, block/interval
times all shifted.

### 15.2 New measure kinds

- **Aperiodic numeric / event / state series** ride the existing block+interval transfer
  path; their block header times are already shifted. The event-pair *derivation* (§14)
  is definition-level, so the exported `DatasetDefinition` must carry the pair request
  (option 3) — or the materialized label set must transfer (option 1, already handled).
- **String value blocks / dictionary codes** transfer as blocks; if the low-road global
  dictionary table (code→string) is used, it must be transferred and remapped like any
  other id table. Dictionary strings are a **de-id surface** — an event/mode vocabulary
  is usually safe, but free-text payloads may contain PHI and need review/scrubbing.

### 15.3 Tables NOT transferred today that should be — with PHI/time fields flagged

| Table | Time fields → **shift** | Identifier/quasi-id fields → **de-id** | Notes |
|---|---|---|---|
| **encounter** | `start_time`, `end_time`, `last_updated` | `visit_number` (**direct identifier — drop or hash**), `patient_id` (remap), `bed_id` (location → quasi-id) | The headline addition; needed for `within: encounter`. Requires bed transfer or bed-drop policy. |
| **device_encounter** | `start_time`, `end_time` | `device_id` (remap), `encounter_id` (remap) | Depends on encounter + device. |
| **bed / unit / institution** | — | `name`s are institutional quasi-identifiers (hospital/unit/bed) | Policy: transfer identified, pseudonymize names, or keep structure only. A rare unit + a date can re-identify even after time-shift. |
| **source** | — | low risk | Transfer for referential integrity (measures/patients/labels carry `source_id`). |
| **log_hl7_adt** | `event_time`, `admit_time`, `discharge_time`, `dob` | `mrn`, names, `visit_num`, `location`, `previous_location`, everything | **Never transferred** (decision) — excluded unconditionally, identified or not. A future opt-in flag may allow it; until then transfer must skip it entirely. |

### 15.4 Recommendations

- Add an `encounter` (+`device_encounter`, +`bed/unit/institution` as needed) transfer
  step with: `patient_id`/`device_id` remap, **`visit_number` dropped or hashed under
  de-id**, all three encounter times shifted, and a bed/location policy switch
  (`identified | pseudonymize | drop`).
- Exclude `log_hl7_adt` from transfer unconditionally (identified or not); leave a hook
  for a future opt-in flag but do not wire it now.
- Extend the field-by-field time-shift wiring to every new time column above; add a test
  that asserts no un-shifted time survives a `time_shift` transfer (there is precedent:
  `sdk/tests/test_transfer_orphan_device_patient.py`).
- Review string dictionary payloads as a de-id surface before enabling free-text
  transfer under de-id.

## 17. Phase 1 implementation spec — string storage (the low road)

> **Status: ✅ implemented, verified & audited (committed `0cd3b4a` on
> `feature/aperiodic-and-text-support`).**
> `sdk/atriumdb/string_dictionary.py` (`MeasureStringDictionary`), string write unified
> into `write_data`/`write_time_value_pairs`, dedicated `get_string_data`, `get_data`
> guard rails behind the single `MeasureStringDictionary.exists` call site. Tests:
> `sdk/tests/test_string_storage.py` 17/17 pass; numeric `time_value` regressions 21/21
> pass (both re-run independently in Docker/SQLite). Read-unification into `get_data`
> left for P3 per §17.8.
>
> **Audited** (independent agent, `sdk/tests/test_string_storage_audit.py` — 30 pass,
> 2 xfail). Core encode/decode/merge/read/concurrency/guard-rails robust. One real defect:
> **mixing numeric + string values on a single measure is silently accepted and corrupts
> readability** — routed to P2, enforced via the `value_type` column (the 2 xfails flip to
> passing when fixed). Non-blocking notes: within a single `write_time_value_pairs` call,
> duplicate timestamps keep the FIRST element (pre-existing numeric behavior, not
> string-specific — document only); `write_data_easy` does not accept strings **by design**
> (§13 — being deprecated, legacy behavior only).

**Goal:** store and retrieve dynamically-sized string values at scale, with **no C
changes and no block-format changes**, by encoding strings as int64 dictionary codes and
reusing the existing int64 write/read path. This is the foundation every later phase
builds on.

**Non-goals for P1:** `signal_kind` schema column (P2), rasterization / iterator
integration (P3), event pairing / queries (P4), transfer of string dicts (P6). P1 must
not regress any numeric path.

### 17.1 The dictionary file

- **Location:** `<dataset_location>/meta/string_dict/measure_<measure_id>.jsonl`
  (per-measure, per §13 decision). Create `meta/string_dict/` lazily on first string
  write.
- **Format:** append-only JSON Lines. **Record index (0-based line number) is the code.**
  Each line is a JSON-encoded string (so embedded newlines / unicode / empty string are
  safe and unambiguous). No header, no rewrite — only appends.
- **Codes:** monotonically increasing from 0, stable forever. int64 in the block.
- **Existence marks a measure as string-typed** for P1 reads (a proper `signal_kind`
  column in P2 supersedes this; keep the check behind one helper so it is a one-line
  swap later).

### 17.2 New module `sdk/atriumdb/string_dictionary.py`

A small class, no SDK/DB dependencies, unit-testable in isolation:

```
class MeasureStringDictionary:
    @classmethod
    def path_for(cls, meta_dir, measure_id) -> Path
    @classmethod
    def exists(cls, meta_dir, measure_id) -> bool
    @classmethod
    def load(cls, meta_dir, measure_id) -> "MeasureStringDictionary"   # reads file (or empty)
    def encode(self, values: Sequence[str]) -> np.ndarray              # int64 codes; APPENDS new strings
    def decode(self, codes: np.ndarray) -> np.ndarray                  # object array of str
    def __len__(self)                                                   # vocabulary size
```

- `encode` builds/extends an in-memory `{str: code}` map, appends genuinely new strings to
  the file, and returns an `int64` array. Appends happen **under a file lock**
  (`filelock`, already an indirect dep via test infra — confirm; else `fcntl`/atomic
  rename) so two writers cannot assign the same code to different strings. Document the
  single-writer expectation, mirroring `write_data`'s existing merge caveat.
- `decode` maps codes → strings; a code ≥ vocabulary size raises a clear error (indicates
  a dict/data mismatch, never expected within one dataset).
- Non-string / non-`str` inputs to `encode` raise a `TypeError` with a clear message.

### 17.3 SDK write path — unify into the existing write methods

**Decision:** do **not** add a separate `write_string_data`. Instead teach the existing
`write_time_value_pairs` (primary) and `write_data` (advanced) to accept string/object
value arrays, because the write side has no polymorphic-return-contract problem — the
only barrier is that strings must become int64 codes *before* the C encode step, and the
value dtype makes string detection trivial at write time.

- In `write_data` ([atrium_sdk.py:800](../../sdk/atriumdb/atrium_sdk.py)), before
  `_resolve_value_types`, detect a string/object value array
  (`value_data.dtype.kind in ('U', 'S', 'O')`) and convert:
  `codes = MeasureStringDictionary.load(meta_dir, measure_id).encode(value_data)` → int64.
  From that point everything proceeds as an ordinary int64 write
  (`raw_value_type=V_TYPE_INT64` → auto `V_TYPE_DELTA_INT64`, `scale_m=1.0, scale_b=0.0`).
  - `_resolve_value_types` today does `V_TYPE_INT64 if issubdtype(integer) else
    V_TYPE_DOUBLE`, so an unconverted string array would silently fall to DOUBLE — the
    conversion must happen *before* it. If the caller passes an explicit numeric
    `raw_value_type` alongside string data, raise a clear error.
  - Accept a Python `list[str]` too (`np.asarray`), matching how callers pass values.
- `write_time_value_pairs` ([atrium_sdk.py:1570](../../sdk/atriumdb/atrium_sdk.py)) needs
  no special logic beyond passing the string/object array through to `write_data`; relax
  its `values.shape`/dtype assumptions so object arrays pass. This is the ergonomic entry
  point — a user writes `"Anesthesia START"` at timestamps with the same call they'd use
  for NIBP.
- **No new interval/merge/encoding logic** — codes are plain int64 values, so block merge,
  duplicate-newest-wins, the interval index, and the smart time-encoding all work
  unchanged. Because the dict is per-measure and append-only with stable codes, codes are
  consistent across writes, so merging two code blocks is valid.
- `meta_dir` is derived from `self.dataset_location` (add a small `self._meta_dir` helper
  if none exists; `meta/` is already used by transfer for `definition.yaml`).

### 17.4 SDK read path — dedicated `get_string_data(...)` for P1

Reads stay on a **dedicated** getter in P1. This is deliberate: decoding codes→strings is
trivial, but `get_data` is also the windowing iterator's rasterization feeder and its
contract is numeric to the core (the `return_nan_filled` float `out` buffer, analog
scaling, and many numeric-assuming consumers). Folding strings into `get_data` *is* the
rasterization problem, so it is an explicit **P3 deliverable** (see §17.8), not P1.

```
def get_string_data(self, measure_id, start_time_n, end_time_n, device_id=None,
                    patient_id=None, time_units=None, sort=True, ...) -> (times, values:str[]):
```

- Call the existing `get_data(..., analog=False)` to get `(headers, times, codes)`.
- `values = MeasureStringDictionary.load(meta_dir, measure_id).decode(codes)`.
- Return `(times, values)` with `values` an object/str ndarray.
- **Guard rails:** if a caller reaches `get_data` for a string measure with
  `return_nan_filled` set or `analog=True`, raise a clear `ValueError` ("string measures
  cannot be NaN-filled / analog-scaled; use get_string_data"). Detecting the string
  measure inside `get_data` uses the single `MeasureStringDictionary.exists` helper
  (swapped for `signal_kind` in P2).

### 17.8 Deferred to P3 (recorded here so P1 leaves clean seams)

- **`get_data` read-unification:** in P3, `get_data` returns object/str arrays for string
  measures and the iterator consumes them via P3 fill rules (carry-forward / presence /
  validity mask) **instead of** `return_nan_filled`. P1 must therefore keep the
  string-measure check behind `MeasureStringDictionary.exists` (one call site) so P3 can
  redirect it without disturbing the numeric path.

### 17.5 Interval index / everything else

Unchanged. A string measure populates `interval_index` exactly like a numeric one
(coarse presence via the branch's widened gap tolerance), so `get_interval_array` already
works for string measures with no P1 change.

### 17.6 Tests (P1 acceptance — a dedicated test session follows)

1. **Round-trip:** write then read identical values, including unicode, empty string,
   very long (> 10 KB) string, embedded newlines/commas/quotes, and repeated values.
2. **Dict append/stability:** a second write mixing old + new strings keeps existing
   codes, appends only new ones, and the `.jsonl` has no duplicate lines.
3. **Persistence:** new `AtriumSDK` object on the same location reads back consistent
   codes/strings.
4. **Merge:** several sub-block-size string writes merge into one block and still
   round-trip (exercises the existing merge path with codes).
5. **Interval index:** `get_interval_array` returns coarse presence for a string measure.
6. **Guard rails:** `return_nan_filled` / `analog=True` on a string measure raise.
7. **Numeric regression:** an existing numeric round-trip test still passes unchanged.
8. *(Optional)* two concurrent `encode` appends under lock produce a consistent dict.

### 17.7 Risks / notes for the P1 worker

- **int64 codes are 8 bytes raw** but delta+zstd crush low-cardinality code streams;
  narrower on-disk codes are the deferred "high road," not P1.
- **Concurrency:** the dict append is the one new shared-state write; the file lock is
  mandatory, and the single-writer-per-measure expectation must be documented next to
  `write_data`'s existing thread-safety caveat.
- **Measure creation** is unchanged — the caller still `insert_measure`s (unit is
  `NOT NULL`; pass a sentinel like `"string"`); P1 does not add schema.
- Keep the "is this measure string-typed?" test behind **one helper**
  (`MeasureStringDictionary.exists`) so P2 can swap it for the `signal_kind` column
  without touching call sites.

## 19. Phase 2 implementation spec — measure metadata + coarse-presence index

> **Status: ✅ implemented & verified (UNCOMMITTED, pending review, on
> `feature/aperiodic-and-text-support`).** Two nullable `measure` columns (`signal_kind`,
> `value_type`) on both backends via an additive idempotent migration mirroring `period_ns`;
> read-time defaults (`NULL`→`waveform`/`numeric`, with a P1 dict-file fallback→`string`);
> idempotent backfill; `insert_measure`/`get_measure_info`/new `get_measure_kind` plumbed;
> `get_data` detection swapped to `value_type` (fallback to `MeasureStringDictionary.exists`);
> and the **P1 mix bug fixed** — `_enforce_value_type_invariant` establishes+persists a
> measure's type on first write and rejects a conflicting kind, so the two audit `xfail`s
> now pass. Tests: dedicated `test_measure_metadata_p2.py` 13/13; string+audit+numeric
> regressions all pass on SQLite; 12 MariaDB write-path tests pass on the new schema
> (all re-run independently). Transfer of the columns/dicts remains a P6 obligation (§19.6).
>
> **Audited** (independent agent, `sdk/tests/test_measure_metadata_p2_audit.py` — 26 tests).
> Confirmed correct: mix-rejection on ALL write entry points, read-time defaults, backfill,
> caching invalidation, concurrency, do-no-harm to transfer, and the **Maria `ALTER` on a
> pre-existing column-less dataset** (closes the earlier residual gap — the auditor dropped
> the columns from a live Maria table and reconnected). One real bug found **and fixed**: a
> write that established+persisted `value_type` on first write *before* downstream validation
> raised could poison a measure (e.g. a rejected `write_data_easy` string call). Fix: split
> into `_check_value_type_invariant` (early, raises on conflict, never persists) and
> `_establish_value_type` (persists only *after* the write commits); the two audit regression
> tests now pass. Also fixed a misleading "period_ns column is missing" message that fired
> for any missing measure column (both backends). Verified: 87 SQLite tests + the Maria
> migration test pass; 84-test numeric regression unaffected.

**Goal:** promote the two independent axes of §4 from *inferred-per-write* to *durable
measure metadata*, so every downstream layer (reads, interval index, later rasterization
and event queries) stops re-guessing; make `get_interval_array` an explicit **coarse
presence** map; and replace P1's file-existence string heuristic with a real column.

**Non-goals:** rasterization / iterator (P3), event pairing & queries (P4),
event-anchored definitions (P5), transfer of the new columns (P6 — but P2 must not make
transfer *worse*; see §19.6).

### 19.1 The key design point to confirm: TWO axes, not one

P1 detected "string measure" by the presence of a dictionary file. §17.8 loosely called
the P2 replacement a "`signal_kind` column" — that was imprecise. **Shape and value type
are independent** (§4):

- **`signal_kind`** — *temporal shape*: `waveform | sample | event | state`.
- **`value_type`** — *value encoding*: `numeric | string`. (Numeric stays int64/double as
  today; `string` means "stored as int64 dictionary codes, decode via
  `MeasureStringDictionary`.")

A string measure can be any shape (an `event` of strings, a `state` of strings, a `sample`
of strings); a numeric measure likewise. So P2 adds **both** columns. String detection
(P1's `MeasureStringDictionary.exists`) becomes `value_type == 'string'`; `signal_kind`
drives presence/rasterization defaults. **✅ CONFIRMED:** two separate columns
(`signal_kind`, `value_type`), not one overloaded column.

### 19.2 Schema change (both backends) + migration

Follow the existing additive pattern — `_column_exists` + `ALTER TABLE measure ADD COLUMN`
guarded at init, exactly as `period_ns` was added
([sqlite_handler.py:161 `update_measure_schema`](../../sdk/atriumdb/sql_handler/sqlite/sqlite_handler.py),
mirrored in `maria_handler.py`):

- `measure.signal_kind TEXT NULL` (sqlite) / `VARCHAR` (maria), values in the enum above.
- `measure.value_type TEXT NULL` / `VARCHAR`, `numeric | string`.
- Add to the `CREATE TABLE measure` in `sqlite_tables.py` / `maria_tables.py` for fresh
  datasets, **and** an idempotent `update_measure_schema`-style migration for existing
  ones.

**Backfill for existing datasets (must be safe on production data, write-once history):**
- `NULL` is treated as the legacy default `signal_kind='waveform'`, `value_type='numeric'`
  everywhere it is read (never rely on the backfill having run — read-time default is the
  source of truth). This keeps every existing dataset correct with zero migration risk.
- One-time opportunistic backfill: any measure that already has a P1 dictionary file
  (`MeasureStringDictionary.exists`) is set `value_type='string'`; `signal_kind` is left
  `NULL`/`waveform` unless inferable. Idempotent, re-runnable.

### 19.3 API plumbing

- `insert_measure` ([atrium_sdk.py:2639](../../sdk/atriumdb/atrium_sdk.py)) +
  `sql_handler.insert_measure`: add optional `signal_kind=None`, `value_type=None`.
  Explicit values win; when omitted, `value_type` is inferred at first write from the
  value dtype (string/object → `string`, else `numeric`) and `signal_kind` defaults to
  `waveform` unless the write classified the data as aperiodic (reuse the
  smart-write-defaults median-spacing classification: `sample`/`event`/`state` require an
  explicit hint in P2 — automatic *shape* inference beyond waveform-vs-aperiodic is out of
  scope; document that `sample` is the safe default aperiodic-numeric kind).
- `get_measure_info` ([atrium_sdk.py:2341](../../sdk/atriumdb/atrium_sdk.py)) returns both
  new fields.
- **Swap the P1 detection site:** `get_data`'s guard now reads `value_type` (falling back
  to `MeasureStringDictionary.exists` when the column is `NULL`, so un-migrated datasets
  still work). This is the single call site P1 kept isolated.
- **Enforce the numeric/string invariant (fixes the P1 audit bug).** On write, once a
  measure's `value_type` is established (explicitly, by the column, or by the P1
  fallback: dictionary file present → string, existing blocks → numeric), reject a write
  whose value-kind conflicts — a numeric write to a string measure, or a string write to a
  numeric measure — with a clear error, instead of silently corrupting readability. Verify
  the two `xfail` tests in `sdk/tests/test_string_storage_audit.py`
  (`test_numeric_then_string_should_be_rejected`, `test_string_then_numeric_should_be_rejected`)
  now pass (remove the `xfail` markers).

### 19.4 Interval index = documented coarse presence

- No structural change to `interval_index` (the smart-write branch already widens the gap
  tolerance for aperiodic writes). P2 makes the *contract* explicit: `get_interval_array`
  returns **coarse presence**, and its docstring says so for `sample/event/state` kinds.
- Optional, low-risk: when the caller does not pass `gap_tolerance_nano`, pick a
  `signal_kind`-aware default on read (waveform → tight, aperiodic → generous) instead of
  `0`, so the returned intervals are sensible per kind. Keep behind the existing param so
  callers can always override.

### 19.5 Direct event/sample time read

Confirm (and document) that `get_string_data` / `get_data` already return the **actual
stored timestamps** for aperiodic kinds — that is the "precise" read that should be used
instead of `get_interval_array` for event/sample logic. Add a thin
`get_measure_kind(measure_id) -> (signal_kind, value_type)` convenience if useful. No new
storage.

### 19.6 Transfer note (do no harm)

P2 does not implement column transfer (that is P6), but `transfer_measures` must not choke
on the new columns and should carry them through when trivially available. If carrying
them through is not trivial, P2 leaves them to default on the destination (correct via the
read-time default) and P6 wires them properly. **✅ CONFIRMED:** defer column transfer to
P6. **P6 obligation (must not be dropped):** `transfer_measures` must carry `signal_kind`
and `value_type`, and the string `meta/string_dict/measure_<id>.jsonl` dictionaries must be
transferred alongside the codes (else string reads on the destination decode to garbage).
This is a hard requirement recorded here so the deferment does not become an omission.

### 19.7 Tests

Fresh-dataset create has the columns; migration adds them to a column-less dataset and is
idempotent; `NULL` reads as `waveform`/`numeric`; a P1 string dataset backfills to
`value_type='string'` and `get_data` still guards correctly via the column; `insert_measure`
round-trips explicit values through `get_measure_info`; numeric and string P1 tests still
pass unchanged.

## 21. Phase 3 implementation spec — rasterize into the Window contract

> **Status: ✅ implemented, audited & bug-fixed (UNCOMMITTED, pending final numeric
> regression, on `feature/aperiodic-and-text-support`).** Per-kind fill rules in
> `windowing_functions.py` (`_rasterize_grid` with a *separable* known-boolean → sentinel,
> per §21.2#2b), `dataset_iterator.py` (nominal-period resolution, `aperiodic_fill`/
> `fill_overrides`/`period_overrides`, decode accessors), `window.py`
> (`decode_string_signal`), `string_dictionary.py` (`UNKNOWN_STRING_CODE = -1` — a negative
> sentinel, since committed dicts already use code 0), and `get_iterator`. Waveform-numeric
> path kept byte-for-byte (two independent equivalence tests + the real MIT-BIH
> `test_iterator.py` regression passed).
>
> **Audited** (independent agent, `sdk/tests/test_aperiodic_windowing_p3_audit.py`). Confirmed
> correct: −1 sentinel never confusable with real codes, no batch-sizing distortion, waveform
> path unchanged. One **HIGH** bug found **and fixed**: carry-forward/left-censoring output
> depended on `num_windows_prefetch` (a RAM knob) because each batch only read its own time
> span, so a reading before a window's batch was invisible and a known cell was emitted as the
> unknown sentinel. Fix: propagate the definition's true range start through the batch tuples
> and **seed each carry-forward batch** from the last reading in `[range_start, batch_start)`
> (bounded lookback, batching intact); `sparse`/`aggregate`/`presence`/`count` unaffected. Also
> replaced an opaque "slice step cannot be zero" error (period > window) with a clear message.
> The 3 audit `xfail` regression tests now pass. Combined P3 suites: **40 pass, no xfail**.
> Because the fix touched core batch construction, the full numeric `test_iterator.py` is being
> re-run as the final gate before commit.
>
> **Known limitations (documented, by phase boundary):** unknown is a *sentinel in values*
> (no separate `known` mask yet — structured to add one later); state **right-censoring** is a
> no-op until P4 pairing; `lightmapped` / `DatasetDefinition.filter` keep the numeric-only path.

**Goal:** make the windowing iterator produce sensible fixed-shape windows for aperiodic
and string measures, using per-`signal_kind` fill rules, a **validity/known mask** for
gaps and censoring, a **1 s default raster period** when a measure has none, and by
**folding string reads into `get_data`** (the read-unification deferred from P1 §17.8).

**Non-goals:** event *pairing* / "in state A→B" derivation (P4 — but P3 must build the
state carry-forward + mask machinery P4 feeds); event-anchored definitions (P5); ragged
(non-rasterized) window output (a possible later mode — P3 is rasterize-only).

### 21.1 Today's iterator (what changes)

`get_signal_dictionary` ([windowing/windowing_functions.py:49](../../sdk/atriumdb/windowing/windowing_functions.py))
builds, per measure, a regular grid `arange(start, end, period_ns)`, NaN-fills a **float**
value array, drops real samples in via `get_data(..., return_nan_filled=out)`, then
`sliding_window_view`s it. `DatasetIterator` uses `lowest_period_ns = min(period_ns)` for
batch sizing. Three things break for aperiodic/string: (a) an aperiodic measure has no
natural grid period; (b) NaN-fill can't hold strings and mis-represents "no reading" vs
"gap"; (c) `get_data` raises for string measures on the NaN-fill path (P1 guard).

### 21.2 Decisions (✅ confirmed with requester)

1. **✅ Grid period = per-measure nominal, default 1 s (reuse `period_ns`, NO new column).**
   Keep the existing *per-measure* grid. The per-measure nominal raster period is the
   existing `measure.period_ns`; when it is absent/unusable (aperiodic kinds) it defaults to
   **1 s**, and is still overridable via `get_iterator`. Resolution order:
   `get_iterator` override → `measure.period_ns` → 1 s. Do **not** add a schema column.
   `lowest_period_ns` / batch sizing must use this resolved nominal period so an aperiodic
   measure can't distort `row_size`.
2. **✅ Unknown cells = sentinel in `values` (NOT a separate mask) — for now, with two
   requirements.** Represent unknown (gap / left-right-censored state) as a sentinel *in
   the value array*: `NaN` for float channels, and a **reserved "unknown" sentinel code**
   for int/string(code) channels (e.g. reserve dictionary code 0 as `"<unknown>"`, or a
   documented negative sentinel — pick one and apply it consistently). **(a)** This
   limitation MUST be documented: a sentinel conflates "unknown/censored" with a genuine
   missing/NaN reading, and every channel needs a reserved sentinel value. **(b)** Write
   the fill code so the *unknown-ness* is computed as a separable internal step (a boolean
   "is this cell known" is produced internally and then applied as a sentinel), so a future
   switch to emitting a real per-signal `known` mask is an additive change, not a rewrite.
3. **✅ Fill config = default-per-kind + per-measure overrides.** `get_iterator` takes an
   `aperiodic_fill=` default plus `fill_overrides={measure_id: rule}`. Defaults:
   `waveform`→existing NaN grid; `sample`→**carry-forward** with options `sparse` /
   `aggregate:{last|mean|min|max}`; `state`→carry-forward **with left-censoring**;
   `event`→**presence** (0/1) with option `count`.
4. **✅ String values in windows = int64 codes + a dictionary accessor.** Windows carry
   compact int64 dictionary codes (tensor-friendly, memory-efficient for batches); provide
   a helper to decode a window's codes to strings on demand. The reserved unknown sentinel
   from (2) lives in the same code space.

### 21.3 Per-kind fill semantics

Per §21.2(2), "unknown" is a sentinel in the value array (NaN for float; a reserved code
for int/string), computed via a separable internal boolean so a real mask can be added
later.

- **waveform (numeric):** unchanged — NaN grid (NaN = no sample).
- **sample (numeric):** default **carry-forward** (each cell = most recent prior reading;
  cells before the first real sample in the window are the unknown sentinel — NaN). `sparse`
  = value only in the nearest cell, NaN elsewhere. `aggregate:*` = reduce multiple readings
  that fall in one cell.
- **state (numeric or string):** **carry-forward** the value in effect at each cell.
  **Left-censoring:** cells before the first observed transition in the window get the
  **unknown sentinel** (NaN or the reserved code) — state is genuinely unknown (recording
  may have begun mid-state), NOT the first observed value. Right-censoring symmetric if a
  state never closes. This is the machinery P4's "in A→B" 0/1 rides.
- **event (numeric or string):** **presence** — cell = 1 if an event occurred in the cell's
  span, else 0 (exactly what `get_label_time_series` produces; §21.4). `count` = number of
  events in the cell. No unknown sentinel here — absence (0) is meaningful.

### 21.4 Reuse the label rasterizer

`get_label_time_series` already rasterizes intervals to a 0/1 presence array over a
timestamp grid ([atrium_sdk.py](../../sdk/atriumdb/atrium_sdk.py)) and the iterator already
consumes it. **Route event-presence and state-membership rasterization through the same
code** rather than a parallel implementation; generalize it to emit the `known` mask.
P4's derived A→B intervals then feed this same path.

### 21.5 `get_data` and the iterator's code path

Because §21.2(4) puts **int64 codes** in the window, the iterator does **not** need
`get_data` to return decoded strings — it reads raw codes via the existing
`get_data(..., analog=False)` path (which already returns int64 codes for a string measure)
and applies the P3 fill rules itself, instead of `return_nan_filled`. Consequences:

- **The iterator restructures `get_signal_dictionary`** to branch by kind: `waveform` keeps
  the existing NaN-fill path; `sample`/`state`/`event` (numeric or string) use the new fill
  path over `get_data(analog=False)` output. The P1 guard that raised on `return_nan_filled`
  for string measures is simply not hit on this path (the iterator no longer nan-fills them).
- **Numeric `get_data` stays byte-for-byte unchanged** — highest regression risk; do not
  touch its numeric behavior. Cover with the existing numeric iterator tests.
- **DEFERRED (not P3 critical path):** folding *decoded string* returns into `get_data`'s
  default (so direct users skip `get_string_data`) is a separable convenience with the
  return-type-polymorphism risk flagged in §5.6/§17.4. It is **not required** for the
  iterator and is deferred; `get_string_data` remains the string reader. Revisit later.

### 21.6 Tests

Per-kind rasterization (waveform/sample/state/event × numeric/string) into windows with the
expected unknown-sentinel placement; carry-forward vs sparse vs aggregate for `sample`;
left-censored `state` yields the unknown sentinel (NaN / reserved code) before the first
transition, NOT a fabricated value; the reserved unknown code is never confused with a real
string; 1 s default period when a measure has none; mixed-rate window (a waveform + an
aperiodic measure in one definition) batches correctly; string windows carry int64 codes
with a working decode accessor; `get_data` string round-trip via the unified path; and —
critically — the existing numeric windowing/iterator tests pass unchanged (numeric `get_data`
byte-for-byte identical).

## 22. Phase 4 implementation spec — event query surface + pairing

> **Status: ✅ implemented, audited & bug-fixed (UNCOMMITTED, pending commit, on
> `feature/aperiodic-and-text-support`).** Standalone SDK methods in `atrium_sdk.py`:
> `get_measure_string_vocabulary` (all values, from the dict file),
> `get_string_values_present` (range-scoped distinct), and `get_event_intervals`
> (COLLAPSE pairing via `searchsorted`+`np.unique`, the `within` cascade
> `device_patient → encounter → whole-stream` forceable + warn-not-drop + runs with an
> empty `device_patient` table, and `start/end_censored` flags always clipped to a real
> boundary). `MeasureStringDictionary` gained `vocabulary()` / `code_for()`. Returns the
> exact interval shape P3's state rasterizer consumes (no second rasterizer).
>
> **Audited** (independent agent, `sdk/tests/test_event_intervals_p4_audit.py`, incl. an
> independent brute-force cross-check over ~1000 randomized sequences). Core verdict:
> collapse pairing, `within` cascade (incl. empty `device_patient` and boundary-spanning
> splits), and censoring all correct. Two low-severity issues found **and fixed**:
> (1) `get_string_values_present` now raises a clear error on a missing range (was a cryptic
> `TypeError`); (2) `get_event_intervals` now rejects `from_value == to_value`. A latent
> same-timestamp case in the `_pair_from_to` helper is documented as a precondition
> (unreachable via the public API, since storage dedups coincident timestamps). Verified:
> combined P4 + audit + string suites **82 pass, no xfail**.

**Goal:** turn stored event/string series into queryable events — (a) enumerate the unique
event types, (b) derive `from → to` intervals (pair an event with the next event), scoped
by a `within` container — as standalone SDK methods that later feed P3's state rasterizer
(0/1 "in A→B") and P5's event-anchored `DatasetDefinition`.

**Non-goals:** event-anchored `DatasetDefinition` regions (P5); transfer (P6). P4 is
standalone read/query methods + the pairing/within engine.

### 22.1 What P4 delivers

1. **Enumerate unique event types.**
   - `get_measure_string_vocabulary(measure_id)` — ALL values ever written, read cheaply
     from the per-measure dictionary file (no data scan). Bounded by vocabulary size.
   - Range/source-scoped distinct values — reads the codes for a source over `[start,end]`
     and uniques them (for "what events occurred for device X last week").
2. **`from → to` interval derivation** —
   `get_event_intervals(measure, source, from_value, to_value, within=..., start, end,
   time_units=...)` → a list of `(start_ns, end_ns, start_censored, end_censored)`. Reads
   the string sample series (int64 codes) for the source over the range, finds the code
   positions of `from_value`/`to_value`, and pairs each `from` with the appropriate `to`
   (§22.2#2), intersected with the `within` container (§22.2#3). Vectorized via
   `searchsorted` on the code array — no per-event Python loop.
3. **The `within` cascade** (confirmed earlier, §13): resolve containment intervals as
   **`device_patient` (if populated) → `encounter` → whole-stream / the definition's own
   range**. Intersect candidate `(from, to)` spans with the container; a pair that would
   span a container boundary is rejected (or clipped + flagged). **Warn, do not silently
   drop**, when the requested scoping data is missing; must run with an empty
   `device_patient` table.
4. **Censoring** (rides P3 semantics): a `from` with no following `to` in range/container →
   right-censored (clip end to container/range end, `end_censored=True`); a `to` with no
   preceding `from` → left-censored (`start_censored=True`). Never fabricate a boundary.

### 22.2 Decisions (✅ confirmed with requester)

1. **✅ Same-measure pairing first.** `from_value` and `to_value` are values within ONE event
   measure (e.g. one "anesthesia_events" stream containing "START" and "STOP"). Cross-measure
   pairing (from in measure A, to in measure B) is a later extension, not P4.
2. **✅ Collapse (first-open → first-close), non-overlapping intervals.** A run of `from`s
   until the next `to` is ONE interval; intervals do not overlap. Matches on/off state
   semantics and is unambiguous. Stack/LIFO nesting can be added later if needed.
3. **✅ `within` = `device_patient → encounter → whole-stream` cascade** (per §13), and a
   caller may force a specific container: `within="device_patient" | "encounter" | "none"`
   (whole-stream) or a named label window. Warn (never silently drop) when the requested
   scoping data is missing; must run with an empty `device_patient` table.
4. **✅ Type enumeration = per-measure dictionary file + range scan; no new SQL table.**
   All-values from the dictionary file (cheap); range-scoped distinct reads the codes over
   the window and uniques them. Defer the optional `string_value_dict` table until a real
   query proves too slow.

### 22.3 How pairing rides P3 (no duplicate rasterizer)

The `(start, end)` intervals `get_event_intervals` returns are the SAME interval shape P3's
`state` rasterizer already turns into a 0/1 membership row (with the unknown sentinel for
censored edges). So "give me a 0/1 signal for whether we're inside Anesthesia START→STOP"
= `get_event_intervals(...)` → feed the intervals to P3's state fill path (P5 wires this
into a `DatasetDefinition`). P4 does not add a second rasterizer; it produces intervals.

### 22.4 Tests

Vocabulary enumeration (dict-file all-values + range-scoped distinct); `from→to` pairing
with the confirmed rule (repeats, back-to-back, a `from` with no `to` → right-censored, a
`to` with no `from` → left-censored); `within` cascade — `device_patient` used when
populated, falls back to `encounter`, then whole-stream when `device_patient` is empty (with
a warning), and a pair spanning a container boundary is rejected/clipped; numeric-measure
inputs rejected with a clear error (events are string measures); vectorized path matches a
brute-force reference on random event sequences.

## 23. Phase 5 implementation spec — event-anchored DatasetDefinition

> **Status: ✅ implemented, audited & bug-fixed (UNCOMMITTED, pending commit, on
> `feature/aperiodic-and-text-support`).** `DatasetDefinition` regions can now be `anchor`
> ("X pre/post around every event occurrence") or `from`/`to` ("between an event and the
> next closing event"), resolved in `verify_definition._resolve_event_region` via P4's
> `get_event_intervals` into the SAME `(start, end)` ranges the existing
> validator/iterator consume — the iterator never learns what an event is. Event measure
> by tag or id (must be string); anchor-only; `within` honored; censoring `clip`+warn
> default (`on_censored` = clip|drop|keep). **Note:** `pre`/`post`/`max_duration` are
> nanosecond integers (consistent with the classic `time0`/`pre`/`post` region keys —
> `time_units` scales only the global bounds); the `5m`/`6h`/`encounter` shorthand in the
> §23.1 YAML above is illustrative, not a supported literal.
>
> **Audited** (independent agent, `sdk/tests/test_event_anchored_definition_p5_audit.py`,
> 42 tests). No hard bugs; `_merge_windows`, `max_duration`, `on_censored`, and classic-
> definition non-regression all verified. Two robustness issues found **and fixed**:
> (1) a tag shared by a numeric + string measure could let the numeric one shadow the
> string one (the value_type-blind "best" rule) — event resolution now **prefers the
> string measure** on a tag; (2) a bogus `within` value silently escaped when a source had
> zero occurrences — `within` is now **validated up front**. Verified: P5 + audit suites
> **64 pass, no xfail**; definition regression **19 pass** on both backends after the
> `verify_definition.py` change.

**Goal:** extend the region-spec vocabulary in `verify_definition._get_validated_entries`
([windowing/verify_definition.py:246](../../sdk/atriumdb/windowing/verify_definition.py))
— today `all` / `{start,end}` / `{time0,pre,post}` — with **event anchors** resolved at
validation time via P4 into concrete `(start, end)` ranges, then intersected with the
source's data union and global bounds exactly like the existing region branches.

**Non-goals:** new storage/query primitives (P4 already provides them); transfer of
event-anchored definitions (P6); the deferred `get_data` string-return convenience.

### 23.1 New region specs

```yaml
device_ids:
  25:
    # (a) X pre / Y post around EVERY occurrence of an event value
    - anchor: "Anesthesia START"
      measure: "anesthesia_events"      # the event (string) measure to look in
      pre:  5m
      post: 5m
    # (b) between an event and the next closing event (P4 get_event_intervals)
    - from: "Anesthesia START"
      to:   "Anesthesia STOP"
      measure: "anesthesia_events"
      within: encounter                 # device_patient | encounter | none (P4 cascade default)
      pre:  0
      post: 0                           # optional padding around each derived interval
      # optional: max_duration: 6h ; on_censored: clip|drop|keep
```

- **`anchor`** → for each occurrence of the event value in `[global bounds]`, emit
  `[t - pre, t + post]`. (Occurrences come from reading the event measure's codes — reuse
  P4's read path.)
- **`from`/`to`** → call `get_event_intervals(measure, from, to, source, within=...)`; each
  returned interval, optionally padded by `pre`/`post` and capped by `max_duration`,
  becomes a region. Censoring flag handling per `on_censored` (§23.2#3).

Both resolve to `(start, end)` lists that flow through the unchanged
`verify_definition → map_validated_sources → DatasetIterator` path.

### 23.2 Decisions (✅ confirmed with requester)

1. **✅ Event measure referenced by tag string (measure id also accepted).**
   `measure: "anesthesia_events"` resolved against the dataset (matches how definitions name
   measures); an int measure id is also accepted.
2. **✅ Anchor-only.** An event region only *defines time ranges*; what each window returns
   comes from the definition's `measures` list. To also get the event channel, the user adds
   the event measure to `measures`. "What to window" and "how to slice time" stay separate.
3. **✅ Censored-anchor default = `clip` + warn**, overridable per region via `on_censored`
   (`clip` | `drop` | `keep`). Clip the censored end to the container/range boundary, keep
   the region, and warn.
4. **✅ `anchor` occurrences honor the region's `within`** (same cascade as `from`/`to`);
   unscoped only when no `within` is given.

### 23.3 Interaction with the rest

- **Validation errors:** an event `measure` tag not found, or a `from`/`to`/`anchor` value
  not in that measure's vocabulary, raises at `validate()` time with a clear message
  (reuse P4's vocabulary checks). An event region on a source with no such event → warns
  and contributes no ranges (consistent with existing empty-source handling).
- **`DatasetDefinition` plumbing:** extend `_check_times_and_warn`
  ([windowing/definition.py](../../sdk/atriumdb/windowing/definition.py)) `allowed_keys`
  with `anchor`/`from`/`to`/`measure`/`within`/`max_duration`/`on_censored`, and the
  region-resolution branch in `_get_validated_entries`.
- **Rasterizing "in A→B" as a 0/1 signal** is already available end-to-end: add the event
  measure (kind `state`/`event`) to `measures`; P3 renders it. P5 is specifically about
  *defining the dataset's time ranges* from events. (Deriving a persisted 0/1 label channel
  from a `from/to` region is a possible later convenience, not P5.)

### 23.4 Tests

`anchor` emits `[t-pre, t+post]` per occurrence and merges/clips against the data union and
global bounds; `from/to` regions match `get_event_intervals` then honor `pre`/`post`/
`max_duration`; `within` on a region scopes correctly (incl. empty `device_patient`);
`on_censored` clip/drop/keep behave; unknown event measure tag or out-of-vocabulary value
raises at validate(); an event region resolves through to the iterator producing the
expected windows; and existing (non-event) definition tests are unchanged.

## 20. Why this fits AtriumDB rather than fighting it

- Text becomes a **value encoding**, so it rides the existing block / index / iterator /
  transfer pipeline instead of a parallel subsystem.
- Aperiodic **numeric** data already fits (this branch); aperiodic **events** fit by
  reusing the block store for payloads and the label tables for curated state.
- Event-anchored cohorts plug into the **existing** definition→validator→iterator flow
  by emitting the same `(start, end)` ranges everything already consumes — the iterator
  never learns what an "event" is.
- SQL gains at most one bounded dictionary table; the write-once/query-many bulk stays
  in immutable columnar blocks.
