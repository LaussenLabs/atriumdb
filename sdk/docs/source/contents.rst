.. atriumdb documentation master file, created by
   sphinx-quickstart on Wed Jan 25 14:44:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API Reference
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: atriumdb
   :members:
   :special-members:

.. autoclass:: atriumdb.AtriumSDK

   .. automethod:: __init__
   .. automethod:: create_dataset

   .. automethod:: insert_measure
   .. automethod:: get_measure_id
   .. automethod:: get_measure_info
   .. automethod:: search_measures
   .. automethod:: get_all_measures

   .. automethod:: insert_device
   .. automethod:: get_device_id
   .. automethod:: get_device_info
   .. automethod:: search_devices
   .. automethod:: get_all_devices
   .. automethod:: get_device_patient_data
   .. automethod:: insert_device_patient_data
   .. automethod:: measure_device_start_time_exists

   .. automethod:: insert_patient
   .. automethod:: get_patient_info
   .. automethod:: get_all_patients
   .. automethod:: get_mrn_to_patient_id_map
   .. automethod:: get_patient_id_to_mrn_map

   .. automethod:: get_all_patient_encounter_data

   .. automethod:: get_interval_array

   .. automethod:: write_data
   .. automethod:: write_data_easy

   .. automethod:: get_data
   .. automethod:: get_iterator

.. autoclass:: atriumdb.DatasetDefinition

   .. automethod:: __init__
   .. automethod:: add_measure
   .. automethod:: add_region
   .. automethod:: save

.. autoclass:: atriumdb.DatasetIterator

   .. automethod:: __len__
   .. automethod:: __getitem__
   .. automethod:: get_signal_window
   .. automethod:: get_array_matrix



Index
--------------------

* :ref:`genindex`
