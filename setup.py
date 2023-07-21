""" geocode-adi setup.py

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
    version="2.1.0",
    description="Map addresses to Area Deprivation Index through U.S. Census Block Groups",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/AyushDoshi/geocode-adi",
    author="Ayush Doshi",
    author_email="ayushdoshi1@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.8',
    packages=["geocode-adi"],
    include_package_data=True,
    install_requires=["requests==2.29.0", "tqdm", "censusgeocode", "geopy", "pandas", "numpy", "pyarrow"],
    package_dir={'/': 'geocode-adi'}
)
