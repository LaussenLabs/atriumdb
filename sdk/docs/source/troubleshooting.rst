.. _troubleshooting:

Troubleshooting
=================

This page addresses common issues encountered when using AtriumDB and provides recommendations for resolving them.

**Contents**

- :ref:`OpenMP Interference <openmp_interference>`
- :ref:`CDLL Not Hashable <cdll_not_hashable>`


.. _openmp_interference:

OpenMP Interference
----------------------

**Problem**

AtriumDB uses OpenMP to facilitate multithreaded encoding and decoding of data blocks.
To control the number of threads OpenMP uses, AtriumDB dynamically sets the environment variable `OMP_NUM_THREADS`
during each read or write operation, based on the `num_threads` parameter provided during `AtriumSDK` initialization.
This behavior can interfere with other libraries that also use OpenMP, such as PyTorch, leading to performance issues or unexpected behavior.

**Solution**

To avoid conflicts, set the `AtriumSDK.num_threads` parameter during initialization to a value suitable for all OpenMP-dependent libraries in your workflow.

For example:

.. code-block:: python

    from atriumdb import AtriumSDK

    sdk = AtriumSDK(dataset_location="./my_dataset", num_threads=4)

**Important Notes:**

- AtriumDB can affect other libraries using OpenMP by dynamically setting `OMP_NUM_THREADS`. Similarly, other libraries that modify this variable can impact AtriumDB’s performance.
- Monitor the overall threading configuration of your application to ensure harmony across all libraries.


.. _cdll_not_hashable:

CDLL Not Hashable
----------------------

**Problem**

AtriumDB (a python library) calls a C shared library (ctypes CDLL or WINDLL) to optimize its CPU-intensive operations.
This shared library is stored as an object variable in the `AtriumSDK` instance upon calling of its `get_data`, `write_data` (or similar) methods.
Due to this, the `AtriumSDK` object becomes unhashable after timeseries data is read or stored, which can cause issues
in parallelized frameworks like multiprocessing, or other hashable use cases.

**Solution**

To ensure the `AtriumSDK` object is hashable, either:

1. Ensure that you do not invoke any timeseries data read or write operations before creating worker processes. This will prevent the shared library from being loaded prematurely.

--or--

2. If you need to use timeseries operations before creating workers, explicitly unset the shared library reference just before creating worker processes (or other hashing functions):

.. code-block:: python

    from atriumdb import AtriumSDK

    sdk = AtriumSDK(dataset_location="./my_dataset")

    # Perform necessary operations that involve timeseries data

    # Unset the shared library reference before worker creation
    AtriumSDK.block.wrapped_dll.bc_dll = None

    # Create worker processes
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        pool.map(worker_function, args_list)

**Important Notes:**

- The shared library is lazy-loaded, meaning it is not loaded until a timeseries read or write operation is performed.
