Guide
=====

.. _installation:

Installation
------------

This project is distributed as a Python package and is hosted on the PyPI package server.
To use `cvi`, first install it using pip:

.. code-block:: shell

   pip install cvi

You can also add the package directly from GitHub to get the latest changes between releases (or from a specific branch) with:

.. code-block:: shell

   pip install git+https://github.com/AP6YC/cvi

Quickstart
----------

This section provides a quick overview of how to use the project.
For more detailed code usage, please see the :ref:`detailed usage` section.

Create a CVI object and compute the criterion value in batch with `get_cvi`:

.. code-block:: python

   # Import the library
   import cvi
   # Create a Calinski-Harabasz (CH) CVI object
   my_cvi = cvi.CH()
   # Load some data from some clustering algorithm
   samples, labels = load_some_clustering_data()
   # Compute the final criterion value in batch
   criterion_value = my_cvi.get_cvi(samples, labels)

or do it incrementally, also with `get_cvi`:

.. code-block:: python

   # Datasets are numpy arrays
   import numpy as np
   # Create a container for criterion values
   n_samples = len(labels)
   criterion_values = np.zeros(n_samples)
   # Iterate over the data
   for ix in range(n_samples):
      criterion_values = my_cvi.get_cvi(samples[ix, :], labels[ix])

.. _detailed usage:

Detailed Usage
--------------

The `cvi` package contains a set of implemented CVIs with batch and incremental update methods.
Each CVI is a standalone stateful object inheriting from a base class `CVI`, and all `CVI` functions are object methods, such as those that update parameters and return the criterion value.

Instantiate a CVI of you choice with the default constructor:

.. code-block:: python

   # Import the package
   import cvi
   # Import numpy for some data handling
   import numpy as np

   # Instantiate a Calinski-Harabasz (CH) CVI object
   my_cvi = cvi.CH()

CVIs are instantiated with their acronyms, with a list of all implemented CVIS being found in the [Implemented CVIs](#implemented-cvis) section.

A batch of data is assumed to be a numpy array of samples and a numpy vector of integer labels.

.. code-block:: python

   # Load some data
   samples, labels = my_clustering_alg(some_data)

.. note::
   The `cvi` package assumes the Numpy **row-major** convention where rows are individual samples and columns are features.
   A batch dataset is then `[n_samples, n_features]` large, and their corresponding labels are '[n_samples]` large.

You may compute the final criterion value with a batch update all at once with `CVI.get_cvi`

.. code-block:: python

   # Get the final criterion value in batch mode
   criterion_value = my_cvi.get_cvi(samples, labels)

or you may get them incrementally with the same method, where you pass instead just a single numpy vector of features and a single integer label.
The incremental methods are used automatically based upon the dimensions of the data that is passed.

.. code-block:: python

   # Create a container for the criterion value after each sample
   n_samples = len(labels)
   criterion_values = np.zeros(n_samples)

   # Iterate across the data and store the criterion value over time
   for ix in range(n_samples):
      sample = samples[ix, :]
      label = labels[ix]
      criterion_values[ix] = my_cvi.get_cvi(sample, label)

.. note::
   Currently only using _either_ batch _or_ incremental methods is supported; switching from batch to incremental updates with the same is not yet implemented.

Implemented CVIs
----------------

The following CVIs have been implemented as of the latest version of `cvi`:

* **CH**: Calinski-Harabasz
* **cSIL**: Centroid-based Silhouette
* **DB**: Davies-Bouldin
* **GD43**: Generalized Dunn's Index 43.
* **GD53**: Generalized Dunn's Index 53.
* **PS**: Partition Separation.
* **rCIP**: (Renyi's) representative Cross Information Potential.
* **WB**: WB-index.
* **XB**: Xie-Beni.
