Working with Datasets
========================

.. toctree::
   :maxdepth: 2

Iterator Usage
---------------------

Often we are interested in working with relatively small windows of data at a time. For visualizing, pre-processing
small amounts of data at once, or when we are training a model.

However the primary way of querying data, `AtriumSDK.get_data  <contents.html#atriumdb.AtriumSDK.get_data>`_ incurs an
overhead cost everytime it is called. This makes it an inefficient means of collecting a large amount of small windows
of data.

For this reason, AtriumDB has a `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method, that
preloads large amounts of data in your RAM, and feeds it to you piece by piece in an iterable Class.

`AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ also does the job of windowing and indexing
your data for you, which makes tasks like training a model much simpler.

Dataset Iterator Example
############################

1. Initialize your SDK instance, connected to an existing dataset:

.. code-block:: python

    from atriumdb import AtriumSDK
    sdk = AtriumSDK(dataset_location=local_dataset_location)

2. Define the measures and patient or device cohorts:

**Example 1: By Patient**

.. code-block:: python

    from atriumdb import DatasetDefinition
    measures = ["MLII"]

    patient_ids = {
       1: "all",
       2: [{"time0": 1682739250000000000, "pre": 500000000, "post": 500000000}],
       3: [{"start": 1690776318966000000, "end": 1690777625288000000}],
       4: [{"start": 1690781225288000000}],
       5: [{"end": 1690787437932000000}],
    }

    definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)

**Example 2: By Device**

.. code-block:: python

   device_ids = {
       1: "all",
       2: "all",
   }

   definition = DatasetDefinition(measures=measures, device_ids=device_ids)

You can also use mrns or device tags to device your sources. See the
`DatasetDefinition Class <contents.html#atriumdb.DatasetDefinition>`_ for more options.

3. Set your desired parameters: **window_duration** and **window_slide** (durations in nanoseconds by default,
changeable using ``time_units`` param, output times with conform to ``time_units`` units):

.. code-block:: python

   slide_size_nano = window_size_nano = 60_000_000_000  # 1 minute nano

4. Optional parameters:

**num_windows_prefetch** is the number of windows to preload for optimization, a higher number
increases efficiency at the cost of RAM usage (default will pick the number of windows such that the total number of
cached values is closest to 10 million values.)

**shuffle** When True, Randomizes the order of the dataset slices and the windows within a slice. You can define a slice size
using **cached_windows_per_source** which is the number of windows you want each slice to contain. Setting ``cached_windows_per_source=1``
ensures true randomness, but at great cost to the speed in which the windows are iterated.

**time_units** defines the time units of ``window_duration``, ``window_slide`` and ``gap_tolerance`` options are
``["s", "ms", "us", "ns"]``, default ``"ns"``.

Check the `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ documentation for a complete list of parameters

.. code-block:: python

   num_windows_prefetch = 100_000  # preload 100,000 windows before emitting
   gap_tolerance = 60  # Fill gaps in data less than 1 minute with nans
   time_units = "s"

5. Obtain the iterator:

.. code-block:: python

   iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano,
        num_windows_prefetch=num_windows_prefetch, gap_tolerance=gap_tolerance, time_units=time_units)

4. Iterate through the dataset:

.. code-block:: python

    for window_i, window in enumerate(iterator):
        print()
        print(window.start_time)
        print(window.device_id)
        print(window.patient_id)
        for (measure_tag, measure_freq_hz, measure_units), signal_dict in window.signals.items():
            print(measure_tag, measure_freq_hz, measure_units, signal_dict['measure_id'])
            print('times', signal_dict['times'])
            print('values', signal_dict['values'])
            print('expected_count', signal_dict['expected_count'])
            print('actual_count', signal_dict['actual_count'])

You can find explanations of the returned Window object in the :ref:`window_format` section below.

.. _window_format:

Window Format
#####################

The ``Window`` class represents a data structure for windowed data output by the
`DatasetIterator Class <contents.html#atriumdb.DatasetIterator>`_, it includes the raw
data organized into signal dictionaries, along with associated metadata, and additional
information related to patient and analysis results.

**Attributes**:

- ``signals`` : ``dict``
    A dictionary where each entry corresponds to a different measure signal, making it easier to handle measures of different frequencies. The keys of this dictionary are tuples, each consisting of the measure tag, the frequency of the measure (in Hz), and the units of the measure. The values are dictionaries containing metadata and data for each measure.

    Each signal dictionary has the following structure:

    - ``times`` : ``np.ndarray``
        A 1D numpy array representing the timestamps corresponding to each data point of the signal. This ensures that each data point in the window is associated with its precise capture time.

    - ``values`` : ``np.ndarray``
        A 1D numpy array containing the actual data points of the signal.

    - ``expected_count`` : ``int``
        An integer representing the expected number of data points in the signal window, based on its frequency and the window size.

    - ``actual_count`` : ``int``
        An integer representing the actual number of non-NaN data points in the signal window.

    - ``measure_id`` : ``int``
        An integer representing the unique identifier of the measure.

- ``start_time`` : ``int``
    The starting time, as a nanosecond epoch, of the window.

- ``device_id`` : ``int``
    An identifier representing the device from which the data was captured.

- ``patient_id`` : ``int``
    An identifier representing the patient associated with the data.

- ``label_time_series`` : ``np.ndarray``
    A 1D numpy array representing the labels for each data point in the window, typically used in supervised learning scenarios.

- ``label`` : ``np.ndarray``
    A 1D numpy array representing the aggregated or final label for the window, used for classification or regression outputs.

- ``patient_info`` : ``dict``
    A dictionary containing static patient meta information (such as id, mrn, gender, dob, etc.) returned by
    `AtriumSDK.get_patient_info`, as well as any dynamic fields requested in the `patient_history_fields` of `AtriumSDK.get_iterator`.
    This may include historical measurements like height and weight, along with their units and the timestamps they were recorded.


Example of the ``signals`` dictionary:

.. code-block:: python

    {
        ('heart_rate', 1, 'bpm'): {
            'times': np.array([1, 2, 3, ...]),
            'values': np.array([70, 71, 69, ...]),
            'expected_count': 100,
            'actual_count': 100,
            'measure_id': 123,
        },
        ('temperature', 0.01, 'C'): {
            'times': np.array([0, 10, 20, ...]),
            'values': np.array([36.6, 36.7, np.nan, ...]),
            'expected_count': 10,
            'actual_count': 9,
            'measure_id': 456,
        }
    }

Example of the ``patient_info`` dictionary:

.. code-block:: python

    {
        'id': 1,
        'mrn': 123456,
        'gender': 'M',
        'dob': 946684800000000000,  # Nanoseconds since epoch for date of birth
        'first_name': 'John',
        'middle_name': 'A',
        'last_name': 'Doe',
        'first_seen': 1609459200000000000,  # Nanoseconds since epoch
        'last_updated': 1609545600000000000,  # Nanoseconds since epoch
        'source_id': 1,
        'height': {  # Dynamic field example
            'value': 50.0,
            'units': 'cm',
            'time': 1609544500000000000,  # Nanoseconds since epoch
        },
        'weight': {  # Dynamic field example
            'value': 10.1,
            'units': 'kg',
            'time': 1609545500000000000,  # Nanoseconds since epoch
        }
    }


Iterator Types
------------------------

The `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method supports three different types of iterators: default, filtered, and mapped. Each type serves different purposes and offers unique functionalities to handle your dataset windows as per your needs.

Default Iterator
####################

By default, or if you set `iterator_type` to None or `"iterator"`, you get the standard iterator. This returns an object that implements the `__iter__` and `__next__` methods, which allows you to iterate over a dataset's windows. The windows returned are :ref:`Window <window_format>` objects, which you can query for relevant signals, start time, device information, patient information, and any labels specified in your :ref:`dataset definition <definition_file_format>`.

Filtered Iterator
####################

The filtered iterator is similar to the default iterator, but it adds an additional filter functionality by accepting a user made filter function that decides whether a window should be included or skipped during iteration.

To use the filtered iterator, set `iterator_type` to `"filtered"` and pass a filter function using the `window_filter_fn` parameter. This filter function should take a window object as input and return `True` if the window should be included and `False` otherwise.

Example of defining a filter function:

.. code-block:: python

    def my_filter(window):
        # Your condition here
        return window.signals[("ECG_II", 500.0, "mV")]['actual_count'] >= 5  # at least 5 non-nan values.

    iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano, iterator_type='filtered', window_filter_fn=my_filter)

.. note::

    The filter function is a good place to do preprocessing. Any modifications made to the window object within the filter function will be retained when the window is passed back through the iterator.


Mapped Iterator
####################

The mapped iterator allows random access to dataset windows by using the `__getitem__` method. This means you can index the iterator directly to get a specific window, which is a useful feature if you need precise control over which windows to access, such as when labeling or visualizing specific windows by their indices.

To use the mapped iterator, set `iterator_type` to `"mapped"`. Be aware that this iterator type might be slower compared to the default iterator, as it cannot take advantage of certain optimizations related to sequential access.

Example of using the mapped iterator:

.. code-block:: python

    iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano, iterator_type='mapped')

    # Access a specific window by index
    window = iterator[5]
    print(window.start_time)

Recommendations
####################

- For most use cases, including model training and general window iteration, the default iterator should suffice. If you need to ensure data randomness for model training, you can set the `shuffle` parameter to `True`.
- Use the filtered iterator when you need to filter or preprocess windows on-the-fly.
- Use the mapped iterator for tasks that require random access to specific windows by their indices. However, note that it may be slower due to the lack of sequential access optimizations.

For further information and options on the `get_iterator` method, `check its section in the API Reference  <contents.html#atriumdb.AtriumSDK.get_iterator>`_.

.. _definition_file_format:

Definition File Format
------------------------------

Source Types
#################

The YAML file defines various source types (``patient_ids``, ``mrns``, ``device_ids``, and ``device_tags``). For each source type, specific sources or entities are identified by unique names.

Time Entries
#################

For each source/entity, you can provide multiple time entries. Each time entry describes a relevant time period or event for that source. There are three types of time specifications:

1. **Interval-based**: This type specifies a continuous interval with a ``start`` and/or ``end`` time.

   - ``start``: The beginning of the interval (nanosecond Unix Epoch Time).
   - ``end``: The end of the interval (nanosecond Unix Epoch Time).

2. **Event-based**: This type specifies an event time and the time period before and after the event.

   - ``time0``: The exact time of the event (nanosecond Unix Epoch Time).
   - ``pre``: Duration before the event (in nanoseconds).
   - ``post``: Duration after the event (in nanoseconds).

3. **All** All available time data can be specified using the ``all`` keyword.

Measures
#################

The ``measures`` section lists the measures you want your Dataset to contain. Each measure can either be:

1. The measure tag, if there is only one measure with that tag.
2. A complete measure triplet which includes:

   - ``tag``: The tag identifying the measure.
   - ``freq_hz`` or ``freq_nhz``: The frequency of the measure in Hertz (floating) or nanoHertz (integer).
   - ``units``: The unit of the measure (e.g., volts, bpm).

Labels
#################

The ``labels`` section lists the names of the labels you want to include in your dataset.


.. code-block:: yaml

   # could be mrns, device_ids or device_tags
   patient_ids:
        12345:
            - start: 1682739200000000000  # nanosecond Unix Epoch Time
                end: 1682739300000000000    # nanosecond Unix Epoch Time
            - time0: 1682739250000000000   # nanosecond Unix Epoch Time
                pre: 500000000               # nanoseconds before the event_time
                post: 500000000              # nanoseconds after the event_time
        67890: all
        11111:
            - start: 1682739200000000000  # Start with no end

   measures:
        - heart_rate
        - tag: ECG
          freq_hz: 62.5
          units: mV
        - tag: ABP
          freq_nhz: 250000000000
          units: mV

    labels:
         - sinus_rhythm
         - atrial_fibrillation
         - noise_artifact


Dataset Definitions
-----------------------------

Creating a DatasetDefinition object
###################################

You can create a `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object in several ways:

1. Reading from an existing YAML file:

   .. code-block:: python

      dataset_definition = DatasetDefinition(filename="/path/to/my_definition.yaml")

2. Creating a definition by specific devices:

   .. code-block:: python

      device_tags = {"tag_1": [{'start': start_time_nano_1, 'end': end_time_nano_1}], "tag_2": [{'time0': event_time_nano_2, 'pre': nano_before_event_2, 'post': nano_after_event_2}]}
      labels = ["atrial_fibrillation", "sinus_rhythm", "noise_artifact"]
      dataset_definition = DatasetDefinition(
            measures=measures,
            device_tags=device_tags,
            labels=labels
        )


3. Creating a definition by specific patients:

   .. code-block:: python

      mrns = {1234567: [{'start': start_time_nano_1, 'end': end_time_nano_1}], 7654321: "all"}
      dataset_definition = DatasetDefinition(measures=measures, mrns=mrns)


Building Dataset Definitions from Intervals
###########################################

You can build a dataset definition directly from the measure or label availability using the :py:meth:`DatasetDefinition.build_from_intervals` class method.

This method analyzes available data (via the SDK) and constructs a time-region map for each source by finding regions where the specified signals or labels exist. This approach supports two key use cases:

1. **Measure-based construction**: Specify one or more measures and choose between:

   - **Union** (default): Includes all time intervals where *at least one* of the measures is available.
   - **Intersection**: Includes only those intervals where *all* requested measures are simultaneously available.

2. **Label-based construction**: Specify one or more labels to construct intervals around their presence in the data.

Additional parameters such as `start_time`, `end_time`, and `gap_tolerance` allow precise control over which intervals are included and how gaps in data are handled.

.. code-block:: python

   # Crease a dataset where ECG and ABP are simultaneously available.
   dataset_definition = DatasetDefinition.build_from_intervals(
       sdk=my_sdk,
       build_from_signal_type="measures",
       measures=["ECG", "ABP"],
       start_time=1735845426 * 10**9,
       end_time=1737236445 * 10**9,
       gap_tolerance=1_000_000_000,
       merge_strategy="intersection"
   )

Validating Dataset Definitions
###############################

Dataset definitions must be validated before use with most SDK-based operations.

Calling :py:meth:`DatasetDefinition.validate` performs the following:

- Confirms the existence of the requested measures and labels.
- Identifies the precise start and end times for which those data exist, intersected with the user-defined time specs (if any).
- Stores the mapping of data sources and validated intervals in a structured dictionary.

SDK methods that consume a DatasetDefinition will automatically trigger this validation (only once), but you may optionally call it manually for performance tuning:

.. code-block:: python

   dataset_definition.validate(
       sdk=my_sdk,
       gap_tolerance=5,
       measure_tag_match_rule="best",
       start_time=1735845426,
       end_time=1737236445,
       time_units="s"
   )

.. note::

   Repeated validation calls are avoided once the dataset is marked as validated internally via the ``is_validated`` flag.

Filtering Dataset Definitions
##############################

Once a DatasetDefinition is validated, you can filter it further using the :py:meth:`DatasetDefinition.filter` method.

Filtering works by sliding a window over the time ranges in the validated dataset. Each window is passed to your filter function, which returns `True` (keep) or `False` (discard). This can be used to exclude time periods with missing data, poor signal quality, insufficient label coverage, etc.

Your filter function receives a :ref:`window_format` object and should return a boolean:

.. code-block:: python

   def my_filter_fn(window):
       # Accept only windows where at least 5 samples of the signal are available
       return window.signals[("ECG", 62.5, "mV")]['actual_count'] > 5

   dataset_definition.filter(
       sdk=my_sdk,
       filter_fn=my_filter_fn,
       window_duration=1,
       window_slide=1,
       time_units='s'
   )

You may also control the inclusion of partial windows and adjust other advanced options like label thresholds or custom patient history fields.

Saving Dataset Definitions
###########################

You can save any DatasetDefinition to disk for later reuse using the :py:meth:`DatasetDefinition.save` method.

There are two supported formats:

- **YAML (.yaml or .yml)**: Saves the raw user-specified dictionary (`data_dict`). This format is suitable for editing or inspection.
- **Pickle (.pkl)**: Saves the fully validated and/or filtered definition, including all computed metadata. This format is faster for reuse and avoids re-validation.

.. code-block:: python

   dataset_definition.save("definition.yaml")  # Save original definition
   dataset_definition.validate(sdk=my_sdk)
   dataset_definition.save("validated_definition.pkl")  # Save validated version


Partitioning Dataset Definitions
################################

The :py:func:`partition_dataset` function is used to split a validated `DatasetDefinition` into multiple partitions—commonly for
training, validation, and testing in machine learning workflows. It ensures that no patient appears in more than one partition,
which helps prevent data leakage across splits. The function also attempts to balance the distribution of selected labels
(defined via `priority_stratification_labels`) proportionally across the partitions according to the provided `partition_ratios`.

The partitioning process is stochastic by default. This means that repeated calls to the function can result in different splits,
even with the same inputs. To improve consistency and reproducibility, you can pass a fixed integer seed via the `random_state` argument.
If you want to explore several possible splits and choose the most balanced one, you can set `n_trials` to perform multiple partitioning attempts.
If `verbose=True`, a summary of each attempt is printed, including label distribution and the `random_state` value used.
You can then reuse the best-performing seed in future runs by passing it to `random_state`.

You can also specify `additional_labels`, which are not used to guide the partitioning but will be included in the partition summary reports.
This can be helpful for evaluating how well these labels are represented in each split.

Example:

.. code-block:: python

   from atriumdb import DatasetDefinition, AtriumSDK, partition_dataset

   definition = DatasetDefinition(
       measures=["ECG", "ABP"],
       device_ids={1: "all", 2: "all"},
       labels=["atrial_fibrillation", "sinus_rhythm"]
   )
   # Validate the definition first
   definition.validate(sdk=my_sdk)

   # Partition into train/val/test using stratified label balancing
   train_def, val_def, test_def = partition_dataset(
       definition,
       sdk=my_sdk,
       partition_ratios=[60, 20, 20],
       priority_stratification_labels=["label1", "label2"],
       random_state=42
   )

   # Or, explore 10 candidate partitions and display the 3 best
   (train_def, val_def, test_def), info = partition_dataset(
       definition,
       sdk=my_sdk,
       partition_ratios=[60, 20, 20],
       priority_stratification_labels=["label1", "label2"],
       n_trials=10,
       num_show_best_trials=3,
       verbose=True
   )


Exporting Datasets
----------------------------

Copying an Existing Dataset
###########################

You can duplicate an entire AtriumDB dataset into a new location with minimal configuration. This is useful for:

- Exporting targeted data for experimentation or research
- Creating backups
- Distributing compute away from production systems
- Migrating data to new hardware or storage environments

By default, `transfer_data` reuses existing waveform blocks (`reencode_waveforms=False`), which is faster and conserves disk I/O.

.. code-block:: python

   from atriumdb import AtriumSDK, DatasetDefinition, transfer_data

   # 1. Open your source and create the export destination
   main_sdk = AtriumSDK(
       dataset_location=main_dataset_location,
       connection_params=main_connection_params,
       metadata_connection_type="mariadb"
   )
   export_sdk = AtriumSDK.create_dataset(
       dataset_location="/export/path"
   )

   # 2. Mirror all devices and select measures of interest
   all_devices = main_sdk.get_all_devices().keys()
   device_ids = {did: "all" for did in all_devices}
   measures = [("MDC_ECG_AMPL_ST_II", 200, "MDC_DIM_MICRO_VOLT")]

   # 3. Build the dataset definition
   full_def = DatasetDefinition(device_ids=device_ids, measures=measures)

   # 4. Copy the data using fast block reuse (no re-encoding)
   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=full_def,
       reencode_waveforms=False
   )

   # ——> All waveform ECGs and metadata are now copied to "/export/path"

.. note::

   Block reuse is preferred in most cases. Use `reencode_waveforms=True` only when:
   - You want to change block size (e.g., `dest_sdk.block.block_size`)
   - The source dataset has very small or fragmented blocks you'd like to consolidate

Exporting a Specific Patient Cohort
###################################

For targeted research or smaller exports, you can define a subset of MRNs or patient IDs:

.. code-block:: python

   # Export only selected patients
   cohort_mrns = [12345, 67890]
   mrns = {mrn: "all" for mrn in cohort_mrns}
   cohort_def = DatasetDefinition(
       mrns=mrns,
       measures=measures
   )

   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       reencode_waveforms=False
   )

   # ——> Only ECGs for patients 12345 and 67890 are copied

De-identification & Time Shifting
#################################

If you're sharing data externally (e.g., for research or compliance), you can enable:

- `deidentify`: Removes patient-level metadata (name, MRN, DOB, etc.) and scrambles patient IDs
- `time_shift`: Uniformly shifts all timestamps (e.g., to obscure real-time of events)

These can be used independently, but are often combined for data privacy.

.. code-block:: python

   # Scramble patient IDs and shift timestamps by 2 hours
   two_hours_s = 2 * 60 * 60
   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       deidentify="patient_mapping_file.csv",
       time_shift=two_hours_s,
       time_units="s"
   )

   # ——> Timestamps are shifted; a CSV maps old → new patient IDs

.. note::

   The mapping file lets you reconcile anonymized patient IDs if needed in the future.

Export Formats & CSV Example
############################

AtriumDB supports multiple export formats:

- `"tsc"` (default): Native AtriumDB format (required for use with `AtriumSDK.get_data` and other AtriumSDK waveform data retrieval)
- `"csv"`: Well known, tabular text based format.
- `"npz"`: Numpy arrays
- `"parquet"`: Tabular binary data.
- `"wfdb"`: Binary waveform format.

CSV is generally discouraged for waveform data due to high sample frequency and volume,
disk usage becomes quickly unscalable, but still can be used for small-scale exports if desired.

.. code-block:: python

   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=full_def,
       export_format="csv",
       reencode_waveforms=False
   )

   # ——> Waveforms saved as CSV files under `/export/path/csv/`

