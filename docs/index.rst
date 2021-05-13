tsipy Documentation
===================

Source code is available `here <https://github.com/roksikonja/tsipy>`_.


.. automodule:: tsipy
    :members:


Installation
------------

.. code:: console

    $ pip install tsipy

Installation for Developers
---------------------------


.. code:: console

    # Clone repository
    $ git clone https://github.com/roksikonja/tsipy.git

    # Setup venv
    $ python -m venv venv
    $ source venv/bin/activate

    $ pip install --upgrade pip
    $ pip install wheel

    # For development
    $ pip install -e .[dev]

    # For documentation
    $ pip install -e .[docs]

Getting Started
---------------

.. toctree::
    :caption: Tutorials
    :maxdepth: 1

    Degradation Correction <notebooks/demo_degradation.ipynb>
    Sensor Fusion <notebooks/demo_fusion.ipynb>
    Local GP and Windows <notebooks/demo_localgp.ipynb>
    Measurement Noise Estimation <notebooks/demo_noise.ipynb>

.. toctree::
    :caption: Experiments
    :maxdepth: 1

    VIRGO SPM dataset <notebooks/exp_spm.ipynb>
    ACRIM1 and HF with LocalGP <notebooks/exp_acrim1_hf.ipynb>

.. toctree::
    :caption: Package Reference
    :maxdepth: 2

    Degradation Correction <correction.rst>
    Sensor Fusion <fusion.rst>
    Utilities <utils.rst>


Licence
-------

tsipy is licensed under the MIT License.

Citation
--------

.. code::

    @misc{kolar2020iterative,
          title={Iterative Correction of Sensor Degradation and a Bayesian Multi-Sensor Data Fusion Method},
          author={Luka Kolar and Rok Šikonja and Lenart Treven},
          year={2020},
          eprint={2009.03091},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
