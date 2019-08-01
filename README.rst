=================================================
Integrative analysis of single cell imaging mass citometry data of breast cancer patients
=================================================

.. image:: https://readthedocs.org/projects/spatial_ops/badge/?version=latest
        :target: http://spatial_ops.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status               

.. image:: https://img.shields.io/travis/DerThorsten/spatial_ops.svg
        :target: https://travis-ci.org/DerThorsten/spatial_ops


.. image:: https://circleci.com/gh/DerThorsten/spatial_ops/tree/master.svg?style=svg
    :target: https://circleci.com/gh/DerThorsten/spatial_ops/tree/master

.. image:: https://dev.azure.com/derthorstenbeier/spatial_ops/_apis/build/status/DerThorsten.spatial_ops?branchName=master
    :target: https://dev.azure.com/derthorstenbeier/spatial_ops/_build/latest?definitionId=1&branchName=master

We perform an integrative analysis of multiplexed proteomics single cell spatial data from breast cancer tissues using deep convolutional variational autoencoders. 

Features
--------
Current features include: 
  * Conda ready
  * pytest unit test
  * continous integration
  * coverall code coverage
  * documentation with Sphinx
  * documentation on Read the Docs


Running a first exploratory data analysis
================
First, install the dependencies with

``conda env create -f spatial_ops-dev-requirements.yml``

and activate the corresponding conda environment

``conda activate spatial-dev``

Now, if you are in DKFZ cluster the data is already present (in ``/icgc/dkfzlsdf/analysis/B260/projects/spatial_zurich/data``) so you can run the exploratory data analysis simply with the command

``snakemake``

If you are not in the cluster you first need to update the code in ``folders.py`` by inserting the path of the root folder of the data in your machine (note that the data is not publically available). In the root folder the data must be organized into this directory tree:

::

    <data_root_folder>/
    ├── csv/
    │   ├── Basel_PatientMetadata.csv
    │   ├── Basel_Zuri_SingleCell.csv
    │   ├── Basel_Zuri_StainingPanel.csv
    │   ├── Basel_Zuri_WholeImage.csv
    │   └── Zuri_PatientMetadata.csv
    ├── Basel_Zuri_masks/
    │   └── *.tiff (746 files)
    └── ome/
        └── *.tiff (746 files)
        
The Data
====

The data, from the B. Bodenmiller lab, is a collection of images acquired with Imaging Mass Citometry of breast cancer cells of different patients and under different conditions [1]_.
Each ``.tiff`` file in the ``ome`` folder is uniquely paired with a ``.tiff`` mask. Each mask tells which are the cells.

FAQ
====

Q: Is the data showing 2D sections of 3D bodies?

A: No

----

.. [1] Schulz D, Zanotelli VRT, Bodenmiller B. et al. *Simultaneous Multiplexed Imaging of mRNA and Proteins with Subcellular Resolution in Breast Cancer Tissue Samples by Mass Cytometry.* Cell Syst. 2018 Jan 24

