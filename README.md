# geocode-adi

## What is it?

**geocode-adi** is a Python package that allows for the mapping of address information to Gopal K. Singh's [**Area
Deprivation Index (ADI)**][adi-paper] at the U.S. Census Block Group level, which was calculated by the 
[**Applied Population Lab at UW-Madison**][pop-lab] and made available on 
[**UW-Madison's Neighborhood Atlas**][neigh-atlas].

## Where to get it?
The source code is currently hosted on GitHub at:
https://github.com/AyushDoshi/geocode-adi.

Binary installers for released versions are available at the 
[Python Package Index (PyPI)][geocode-adi-pypi].
```
pip install geocode-adi
```

### Dependencies
- [Pandas](https://pandas.pydata.org/) - Required for reading in address data, manipulating it, and exporting it back 
out.
- [Numpy](https://www.numpy.org) - Required for chunking address data and adding "Not A Number" values.
- [Requests](https://requests.readthedocs.io) - Required for making GET requests to Google.
- [Geopy](https://geopy.readthedocs.io) - Required for making geocoding API calls to Nominatim/OpenStreetMaps.
- [Census Geocode](https://github.com/fitnr/censusgeocode) - Required for making geocoding API calls to the [U.S. 
Census Geocoder][us-geocoder].
- [tqdm](https://tqdm.github.io/) - Required for making progress bars.

If **geocode-adi** is installed using a binary installer, such as through [PyPI][geocode-adi-pypi], the required 
dependencies should automatically be installed.

## How to use it?
This package is meant to be run as a script in the command-line/terminal, although the individual functions found in the
[*geocodeadi.py*](geocodeadi/geocodeadi.py) file may be imported as a module.

The script takes in a comma delimited file, such as a .CSV, that contains the address information. Specifically, the 
file must contain at least 4 columns: *'Address'*, *'City'*, *'State'*, *'ZIP Code'*. These columns must also be 
labeled as such. The *'Address'* column must contain at least the street number and name. Other information, such as 
apartment or suite number, is optional. An example of a correctly formatted file of addresses is the
[*AddressSample.csv*](AddressSample.csv) file in the repository.

Once installed, **geocode-adi** can be simply called in the command-line/terminal in the following way:
```
python -m geocode-adi [PATH_TO_ADDRESS_FILE]
```
For example, using the file [*AddressSample.csv*](AddressSample.csv):
```
python -m geocode-adi AddressSample.csv
```

## How does it works?

**geocode-adi** works in mapping addresses to ADI in 4 overarching steps:
1. It imports the addresses into a Pandas DataFrame.
2. It converts the addresses into U.S. Census Block Groups, which itself occurs in 6 steps:
    1. It filters out the majority of P.O. and Route boxes from the address list and puts them to the side.
    2. Then it does a first-pass conversion of the filtered addresses directly to U.S. Census Block Groups using the 
    [U.S. Census Geocoder][us-geocoder].
    3. The addresses that failed to be directly converted to U.S. Census Block Groups are then tried to be converted to 
    coordinates using Google.
    4. The addresses that failed at being converted to coordinates by Google are then tried to be converted to 
    coordinates using [Nominatim/OpenStreetMaps](https://nominatim.org/).
    5. The addresses that were successfully converted to coordinates, either by Google or Nominatim/OpenStreetMaps, are 
    then tried at a second-pass conversion to U.S. Census Block Groups with the U.S. Census Geocoder again, but this 
    time using the coordinates instead of the addresses directly.
    6. Addresses that were successfully converted to U.S. Census Blocks, either in the first or second pass, are then 
    combined. Addresses that failed to be converted, due to being a PO/Route Box, failing to be converted to 
    coordinates, or failing to have their coordinates converted to U.S. Census Block Groups, are also combined.
3. Addresses that were successfully converted to U.S. Census Block Groups then have an ADI value mapped to it based on 
its U.S. Census Block Group.
4. The addresses that were mapped to ADI values through their U.S. Census Block Group are exported to a .CSV file 
labeled ***successful.csv***. Addresses that failed to be converted to Census Block Groups are exported to a .CSV file 
labeled ***failed.csv***.

## License
[MIT](LICENSE)

[adi-paper]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1447923/
[neigh-atlas]: https://www.neighborhoodatlas.medicine.wisc.edu/
[geocode-adi-pypi]: (https://pypi.org/project/geocode-adi)
[pop-lab]: https://apl.wisc.edu/
[us-geocoder]: https://geocoding.geo.census.gov/