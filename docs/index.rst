..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home page <self>
   Examples <tutorials>
   API reference <_autosummary/qualitylib>

Welcome to QualityLIB!
======================

QualityLIB is a library that simplifies running quality assessment experiments on video datasets in Python. QualityLIB library interfaces with the `VideoLIB <https://github.com/abhinaukumar/videolib>`_ package to provide an easy API that simplifies quality assessment research tasks such as

#. Specifying and reading datasets of videos, conforming to various ITU standards.
#. Standardizing the implementation of quality models using the :obj:`~qualitylib.feature_extractor.FeatureExtractor` class.
#. Simplifying the execution of feature extraction over datasets using the :obj:`~qualitylib.runner.Runner` class.
#. Standardizing the results of quality modeling using the :obj:`~qualitylib.result.Result` class.
#. Easy interfacing with Scikit-Learn regressor models for routines such as :obj:`~qualitylib.cross_validation`.