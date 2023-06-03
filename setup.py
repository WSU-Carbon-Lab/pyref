#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
]

test_requirements = []

setup(
    author="Harlan Heilman",
    author_email="Harlan.Heilman@wsu.edu",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A simple python package used to analyze X-ray reflectivity data.",
    entry_points={
        "console_scripts": [
            "prsoxs_xrr=prsoxs_xrr.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="prsoxs_xrr",
    name="prsoxs_xrr",
    packages=find_packages(include=["prsoxs_xrr", "prsoxs_xrr.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/HarlanHeilman/prsoxs_xrr",
    version="0.1.0",
    zip_safe=False,
)
