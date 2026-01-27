# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/hydrodepthml
# =============================================================================

from setuptools import setup

# Read the README file for long description

setup(
    name="hydrodepthml",
    version="0.1.0",
    description=("Machine learning model for predicting water "
                 "table depth in crystalline basement aquifers"),
    author="Les Solutions Géostack, Inc",
    author_email="info@geostack.ca",
    url="https://github.com/geo-stack/hydrodepthml",
    license="MIT",
    packages=['hdml'],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta:",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        ],
    )
