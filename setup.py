""" geocodeadi setup.py

Author: Ayush Doshi
"""

import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="geocode-adi",
    version="1.0.0",
    description="Map addresses to Area Deprivation Index through U.S. Census Block Groups",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/AyushDoshi/geocode-adi",
    author="Ayush Doshi",
    author_email="ayushdoshi1@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
    packages=["geocodeadi"],
    include_package_data=True,
    install_requires=["requests", "tqdm", "censusgeocode", "geopy", "pandas", "numpy", "pyarrow"],
    entry_points={
        "console_scripts": [
            "geocode_adi=geocodeadi.__main__:main",
        ]
    },
)
