import argparse
import os

from .geocodeadi import addresses_to_adi


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Address File: Given file, {path}, does not exist.")


parser = argparse.ArgumentParser(description='Convert Addresses to Area Deprivation Indicies.')
parser.add_argument('addresses_file', metavar='Address File', type=file_path,
                    help="Path to file that contains at least 4 columns, 'Address', 'City', 'State', 'ZIP Code', "
                         "and must be labeled as such. 'Address' must contain at least the street number and street "
                         "name; other information, such as apartment/suite number, are not necessary but tolerated.")
parser.add_argument('-sf', '--skip_first_pass', dest='skipfirst', default=False, action='store_true',
                    help='Skip the first pass conversion of addresses to U.S. Census Block Groups using the U.S. '
                         'Census Geocoder. Go straight to addresses to coordinates conversion.')
args = parser.parse_args()

if __name__ == '__main__':
    addresses_to_adi(args.addresses_file, not args.skipfirst)
    print(args.addresses_file, not args.skipfirst)
