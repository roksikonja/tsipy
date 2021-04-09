tsipy Documentation
===================

.. automodule:: tsipy
    :members:

Installation
------------

.. code:: console

    $ git clone https://github.com/roksikonja/tsipy.git
    $ pip install -e .


Getting Started
---------------

.. toctree::
    :caption: Tutorial
    :maxdepth: 1

    Degradation Correction <notebooks/demo_degradation.ipynb>
    Sensor Fusion <notebooks/demo_fusion.ipynb>
    Local GP and Windows <notebooks/demo_localgp.ipynb>

.. toctree::
    :caption: Package Reference
    :maxdepth: 1

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
