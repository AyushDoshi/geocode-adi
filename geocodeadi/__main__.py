"""Address to Area Deprivation Index Geocoder

This script allows the user to convert a list addresses to Area Deprivation Indices through U.S. Census Block Groups.

The tool accepts a comma delimited file (e.g. .csv) that contains at least 4 columns: Address, City, State, and ZIP
Code. The Address column must include at least the street number and street name. Other address information, such as
apartment or suite number if applicable, is optional.

It is recommend that the tool be called as a script from the command-line/terminal.

Author: Ayush Doshi
"""

import argparse
import os

from .geocodeadi import addresses_to_adi


def file_path(path):
    """
    Checks to see if the data file path is valid.

    :param path: Path to data file.
    :type path: str

    :return: Path to data file.
    :rtype: str
    """
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"address_file: Given file, {path}, does not exist.")


def main():
    """
    Main function that is called from command-line/terminal.
    """
    parser = argparse.ArgumentParser(
        description="Convert Addresses to Area Deprivation Indices."
    )
    parser.add_argument(
        "addresses_file",
        metavar="address_file",
        type=file_path,
        help="Path to file that contains at least 4 columns, 'Address', 'City', 'State', 'ZIP Code', "
             "and must be labeled as such. 'Address' must contain at least the street number and street "
             "name; other information, such as apartment/suite number, are not necessary but tolerated.",
    )
    parser.add_argument(
        "-sf",
        "--skip_first_pass",
        dest="skip_first",
        default=False,
        action="store_true",
        help="Skip the first pass conversion of addresses to U.S. Census Block Groups using the U.S. "
             "Census Geocoder. Go straight to addresses to coordinates conversion.",
    )
    args = parser.parse_args()

    addresses_to_adi(args.addresses_file, not args.skip_first)


if __name__ == "__main__":
    main()
