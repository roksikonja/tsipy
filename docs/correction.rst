Degradation Correction
======================

.. module:: tsipy.correction

.. contents::


Exposure
--------

.. currentmodule:: tsipy.correction

.. autofunction:: compute_exposure


Algorithms
----------

.. currentmodule:: tsipy.correction

.. class:: History

    A :func:`~collections.namedtuple` representing signals and ratio at a particular
    step of the iterative degradation correction process.

    .. attribute:: iteration

    .. attribute:: a

    .. attribute:: b

    .. attribute:: ratio


.. autofunction:: correct_degradation

.. autofunction:: correct_one

.. autofunction:: correct_both


Degradation Models
------------------

.. currentmodule:: tsipy.correction

.. autofunction:: load_model

.. autoclass:: DegradationModel
    :members:

.. autoclass:: ExpModel
    :members:

.. autoclass:: ExpLinModel
    :members:

.. autoclass:: MRModel
    :members:

.. autoclass:: SmoothMRModel
    :members:


Signal Generator
----------------

.. currentmodule:: tsipy.correction

.. autoclass:: SignalGenerator
    :members:


References
----------

.. bibliography:: references.bib
   :style: unsrt
