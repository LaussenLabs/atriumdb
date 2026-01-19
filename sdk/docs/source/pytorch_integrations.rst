Pytorch Integrations
====================

Pytorch Dataloader Usage
------------------------


=============================================================
End-to-End Example: Curate, Split, Validate, and Load Dataset
=============================================================
The example shown below is for an ECG multiclass classification problem and will show you all the steps need to get going.

This example shows how to:

#. Connect to AtriumDB and curate a dataset from labeled intervals
#. Split the dataset into train/validation/test **by patient**
#. Validate and filter windows (e.g., enforce majority-label windows)
#. Use a PyTorch :class:`~torch.utils.data.DataLoader` with
   :class:`AtriumDBMapDataset` for model training

.. note::

   All filesystem paths and identifiers below are **generic**. Replace them with
   locations that make sense in your environment. The label names are example
   ECG classes; swap them for your own labels.

Prerequisites
-------------

- ``atriumdb`` Python SDK
- ``torch`` and ``torchvision`` (optional, for using :class:`DataLoader`)
- ``numpy``
- Data and labels in AtriumDB

.. code-block:: bash

   pip install torch numpy atriumdb

Example Setup
-------------

.. code-block:: python

   import os
   import numpy as np
   from datetime import datetime
   from atriumdb import AtriumSDK, DatasetDefinition, partition_dataset

   # --------------------------
   # Generic configuration
   # --------------------------
   DATASET_ROOT = "/path/to/atriumdb_dataset"
   OUTPUT_DIR   = "./dataset_defs"        # where to save YAML/PKL metadata
   os.makedirs(OUTPUT_DIR, exist_ok=True)

   # Labels available in your AtriumDB dataset (example ECG labels)
   LABELS = ["junctional_rhythm", "sinus", "noise", "other"]
   NUM_CLASSES = len(LABELS)

   # Time units for windowing and gap tolerances (example uses *nanoseconds*)
   TIME_UNITS = "s"
   GAP_TOLERANCE = 1      # 1 second tolerance for gaps
   WINDOW_DURATION = 5    # 5-second windows
   WINDOW_SLIDE    = 5    # non-overlapping in this example

   # Sampling configuration (example ECG lead at 500 Hz)
   MEASURE_NAME = "MDC_ECG_ELEC_POTL_II"
   MEASURE_FREQ   = 500
   MEASURE_UNIT = "MDC_DIM_MILLI_VOLT"

   # Optional: patients you want to exclude (e.g., data quality issues)
   EXCLUDE_PATIENTS = set()  # replace with a set of patient IDs to exclude

1) Connect to AtriumDB and Enumerate Patients
---------------------------------------------

.. code-block:: python

   sdk = AtriumSDK(dataset_location=DATASET_ROOT, num_threads=2)

   # Collect all patients present in this dataset
   all_patients = set(sdk.get_all_patients().keys())
   patients = all_patients - EXCLUDE_PATIENTS

2) Build a DatasetDefinition from Labeled Intervals
---------------------------------------------------

We construct a dataset from the labels in atriumdb and add the measure(s) you need.

.. code-block:: python

   data_def = DatasetDefinition.build_from_intervals(
       sdk,
       build_from_signal_type="labels",
       labels=LABELS,
       patient_id_list=patients,
       gap_tolerance=GAP_TOLERANCE
   )

   # Add one or more measures you need for training/inference
   data_def.add_measure(MEASURE_NAME, MEASURE_FREQ, MEASURE_UNIT)

3) Split Into Train/Validation/Test (by Patient) and save YAML dataset definition files
---------------------------------------------------------------------------------------

Split into non overlapping sets **by patient** to avoid leakage across training and evaluation sets.
Here the priority stratification labels tries to distribute that label according to your defined percentages.
With the additional_labels being secondary. This is important because most patients will have multiple labels
and all their labels will have to be in one set so this defines which ones are important to adhere to the split
percentages and which are secondary. Verbose will output the results in a structures format.

.. code-block:: python

   (train_def, val_def, test_def), duration_info = partition_dataset(
       data_def,
       sdk,
       partition_ratios=[60, 20, 20],
       priority_stratification_labels=["junctional_rhythm"],
       additional_labels=["sinus", "noise", "other"],
       verbose=True,
       random_state=42,
       n_trials=1000,
       num_show_best_trials=5,
       gap_tolerance=GAP_TOLERANCE,
   )


   today = datetime.today().strftime("%Y-%m-%d")
   train_yaml = os.path.join(OUTPUT_DIR, f"dataset_def_train-{today}.yaml")
   val_yaml   = os.path.join(OUTPUT_DIR, f"dataset_def_val-{today}.yaml")
   test_yaml  = os.path.join(OUTPUT_DIR, f"dataset_def_test-{today}.yaml")

   train_def.save(train_yaml, force=True)
   val_def.save(val_yaml, force=True)
   test_def.save(test_yaml, force=True)


4) Validate and Filter Windows
------------------------------

This step is where we decide what data we want in the final dataset. Since labels all represent a section of time they are overlayed
on the data which means until you iterate over the dataset you wont know exactly which windows your getting. For example
the window boundary may fall in such a way as the first 2 seconds of the window are sinus the 3rd second has no label and
the last 2 seconds are junctional rhythm. Now you have to decide what you want to do with that window. Do you want to keep it
or exclude it? This is what this step allows you to do. You define a function that takes a window as a param and outputs true
if you want to keep it and false if you don't.

Below is an example filter that:

- Rejects windows with no signal or no labels
- Requires the **dominant** label to be:
  - at least 80% of all labeled points in the window, and
  - present in at least 70% of the window’s available samples

.. code-block:: python

   # this global variable is for if you want to calculate class weights to use in weighted loss function
   global_label_count = np.zeros(NUM_CLASSES)

   def majority_label_filter(window):
       """
       window.label_time_series        -> shape: (num_labels, T), values in {0,1}
       """
       sig = window.signals[(MEASURE_NAME, MEASURE_FREQ, MEASURE_UNIT)]

       if sig.get("actual_count", 0) == 0:
           return False

       label_sums = np.sum(window.label_time_series, axis=1)  # (num_labels,)
       total_labeled = label_sums.sum()
       if total_labeled == 0:
           return False

       dominant_idx = int(label_sums.argmax())
       frac_of_labeled = label_sums[dominant_idx] / max(total_labeled, 1)
       frac_of_window  = label_sums[dominant_idx] / max(sig["actual_count"], 1)

       if frac_of_labeled < 0.80:
           return False
       if frac_of_window < 0.70:
           return False

       global_label_count[dominant_idx] += 1
       return True

   # Validate and filter each split, and also persist a .pkl file to save it
   for name, d in (("train", train_def), ("val", val_def), ("test", test_def)):
       d.validate(sdk, gap_tolerance=GAP_TOLERANCE, time_units=TIME_UNITS)
       d.filter(
           sdk=sdk,
           filter_fn=majority_label_filter,
           window_duration=WINDOW_DURATION,
           window_slide=WINDOW_SLIDE,
           time_units=TIME_UNITS
       )
       d.save(os.path.join(OUTPUT_DIR, f"{name}_dataset_def.pkl"), force=True)

       # if its the training set compute class weights (OPTIONAL)
       if name == "train":
             class_weights = global_label_count.sum() / (global_label_count * NUM_CLASSES)
             print(f"Class weights: {class_weights}")

             with open(os.path.join(OUTPUT_DIR, f"train_pos_weights-{today}.txt"), "w") as f:
                f.write(" ".join([f"{w:.6f}" for w in class_weights]))

       print(f"[{name}] class label counts: {global_label_count}")
       global_label_count[:] = 0  # reset between splits


5) Use a PyTorch DataLoader via AtriumDBMapDataset
--------------------------------------------------

Here we are going to subclass the AtriumDBMapDataset class that's built into AtriumDB to make it more specific to our needs.
AtriumDBMapDataset is a subclass of Pytorch's Dataset class to allow for integration with dataloaders.
You should override the __getitem__() method and include code to preprocess labels, and preprocess your data. Optionally
you can include a collator_fn to specify batching

.. code-block:: python

   import torch
   from atriumdb.pytorch_integrations import AtriumDBMapDataset

   class AtriumDBClassificationDataset(AtriumDBMapDataset):

       def __init__(self, dataset_location, dataset_definition_path, window_duration, window_slide,
                    time_units, memcache_metadata, gap_tolerance, num_threads):

            super().__init__(dataset_location=dataset_location,
                         dataset_definition_path=dataset_definition_path,
                         window_duration=window_duration,
                         window_slide=window_slide,
                         time_units=time_units,
                         memcache_metadata=memcache_metadata,
                         gap_tolerance=gap_tolerance,
                         num_threads=num_threads)

       def __getitem__(self, index):
           return self.iterator_base_instance[index]

       def __getitem__(self, index):
            # ALWAYS keep this here
            window = self.iterator_base_instance.__getitem__(index)

            # you will need to preprocess the label to decide what label you want to give to this window
            self.process_label(window)

            ecg_signal = window.signals[('MDC_ECG_ELEC_POTL_II', 500, 'MDC_DIM_MILLI_VOLT')]
            X = ecg_signal['values']

            # find the indices of the nan values
            nan_indices = np.isnan(X)
            nan_sum = np.sum(nan_indices)

            # this step interpolates across any missing values
            if nan_sum > 0:
                # make an x-axis array the same length as the abp_values
                indices = np.arange(X.size)
                # interpolate across the nan values
                X[nan_indices] = np.interp(indices[nan_indices], indices[~nan_indices], X[~nan_indices])

            # normalize
            if np.std(X) > 0:
                X = (X - np.mean(X)) / np.std(X)
            else:
                X = X - np.mean(X)

            # put your data into a tensor
            data = torch.from_numpy(X[:2500])
            one_hot_label = torch.from_numpy(window.label)

            return [data, one_hot_label]

       # this function preprocesses the label time series into a one-hot-encoded label. This is a multi-class problem
       # in this example so only one label can be true at a time.
       def process_label(self, window):

           # in the label time series each one will have 1's where the label is present and 0's where it's not.
           # so add up each of the time series so we know how many data points have each label
           label_sums = np.sum(window.label_time_series, axis=1)

           # need to check what is the majority label
           dominant_label_idx = label_sums.argmax()

           # make one hot encoded label
           label = np.zeros(len(window.label_time_series))
           label[dominant_label_idx] = 1

           # create a new object attribute to store the preprocessed label so we can access it later
           window.label = label

Now, load the training data and wrap it in a :class:`DataLoader`:

.. code-block:: python

   from torch.utils.data import DataLoader

   train_ds = AtriumDBClassificationDataset(
       dataset_location=DATASET_ROOT,
       dataset_definition_path=train_yaml,
       window_duration=WINDOW_DURATION,
       window_slide=WINDOW_SLIDE,
       time_units=TIME_UNITS,
       memcache_metadata=True,
       gap_tolerance=GAP_TOLERANCE,
       num_threads=4
   )

   train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

   for batch_idx, batch in enumerate(train_loader):
       # 'batch' contains a batch of data from AtriumDB
       # Integrate with your model's forward pass here.
       pass

.. tip::

   - ``num_workers`` in :class:`DataLoader` parallelizes batching at the PyTorch level,
     which is distinct from the ``num_threads`` you allocate to AtriumSDK. This will GREATLY increase speed and should
     be used to buffer out the time the SDK needs to decompress data. This will allow you to saturate the GPU and make
     data retrieval not the rate limiting step.


Appendix: What to Replace in Your Environment
---------------------------------------------

- ``DATASET_ROOT``: your AtriumDB store path/URI
- ``OUTPUT_DIR``: where to write dataset definitions
- ``LABELS``: your task-specific labels
- ``MEASURE_NAME``, ``MEASURE_FREQ``, ``MEASURE_UNIT``: your signal(s) of interest
- ``TIME_UNITS`` and durations: match the time base your AtriumDB store uses
- ``EXCLUDE_PATIENTS``: any set of patient IDs to exclude (optional)