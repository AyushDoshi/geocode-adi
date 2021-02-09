"""Address to Area Deprivation Index Geocoder Functions

This file contains the functions that are used by the __main__.py file of this module to convert addresses to Area
Deprivation Indices.

Although functions may be imported as a module, it is recommended that the tool is as a script used from the
command line/terminal and the path to the data file is provided there.

Author: Ayush Doshi
"""


import concurrent.futures
import importlib.resources
import math
import re
import warnings

import censusgeocode
import geopy
import geopy.exc
import geopy.extra.rate_limiter
import numpy
import pandas
import requests
import tqdm

# Dictionary that contains pairs of benchmarks and vintages needed to acquire U.S. Census Block Group information from
# addresses and coordinates.
benchmark_vintage_pairs_dict = {
    'Public_AR_Current': ['Census2010_Current', 'ACS2019_Current', 'ACS2018_Current', 'ACS2017_Current'],
    'Public_AR_TAB2020': ['ACS2019_TAB2020', 'ACS2018_TAB2020', 'ACS2017_TAB2020', 'Census2010_TAB2020'],
    'Public_AR_Census2020': ['Census2010_Census2020']}

# Convert dictionary of benchmark and vintage pairs to a list of tuples.
benchmark_vintage_tuple_list = [(benchmark, vintage)
                                for benchmark, vintages in benchmark_vintage_pairs_dict.items()
                                for vintage in vintages]

# Ignore FutureWarnings in regards to tqdm Panel. TODO: Fix tqdm panel concerns to remove filter warnings
warnings.simplefilter('ignore', FutureWarning)
# Remove Pandas' chain assignment warning as it is false positive hit.
pandas.options.mode.chained_assignment = None


def addresses_to_adi(address_path, first_pass):
    """
    Maps Area Deprivation Indices to addresses through U.S. Census Block Groups. This is function is broken into 4
    sub-functions.

    :param address_path: Path to the comma delimited file that contains address information, specifically in the format
    4 columns: Address, City, State, and ZIP Code.
    :type address_path: str

    :param first_pass: Boolean that is used to direct whether the first-pass conversion using the
    addresses directly by the U.S. Census Geocoder should be done. The default value is True and is set to False with
    the --skip_first_pass flag.
    :type first_pass: bool
    """
    # 1. Read in the address data and separate the addresses from other patient data.
    addresses_df, patient_data_df = read_in_data(address_path)

    # 2. Convert the addresses to U.S. Census Block Groups, separating addresses that were successfully
    # converted from those that were not.
    addresses_blockgroup_df, failed_addresses_df = addresses_to_blockgroup(addresses_df, first_pass)

    # 3. Map Area Deprivation Indices to addresses using the U.S. Census Block Groups.
    addresses_blockgroup_adi_df = blockgroup_to_adi(addresses_blockgroup_df)

    # 4. Recombine the address data along with associated Area Deprivation Indices with patient data and export to CSV.
    export_data(addresses_blockgroup_adi_df, failed_addresses_df, patient_data_df)


def read_in_data(address_path):
    """
    Reads in the address data, separates the address information from the patient data, and
    uniquifies the addresses while creating a column of list of original indexes that will later be used to map address
    data and patient data.

    :param address_path: Path to the comma delimited file that contains address information, specifically in the format
    4 columns: Address, City, State, and ZIP Code.
    :type address_path: str

    :return: (DataFrame that contains address information, DataFrame that contains patient information)
    :rtype: tuple
    """
    # Read address csv into a Pandas DataFrame and duplicate the indexes into a column called 'data_id'. This column
    # will be used to map addresses at the end.
    print('Reading in address data...')
    df = pandas.read_csv(address_path, low_memory=False, dtype=str) \
        .reset_index().rename(columns={'index': 'data_id'})

    # Raise ValueError if the columns 'Address', 'City', 'State', and 'ZIP Code' are not present in the DataFrame.
    print('Check if the address data has correct columns...')
    if not all(column in list(df.columns) for column in ['Address', 'City', 'State', 'ZIP Code']):
        raise ValueError("The address data is missing some or all of the following columns: "
                         "'Address', 'City', 'State', and 'ZIP Code'.")

    # Separate the initial DataFrame into one with address information and another with patient information, with
    # 'data_id' being the relational key.
    print('Separating addresses from other patient data...')
    addresses_df = df[['data_id', 'Address', 'City', 'State', 'ZIP Code']]
    patient_data_df = df.drop(columns=['Address', 'City', 'State', 'ZIP Code'])

    # Uniquify addresses and create a new column that contains a list of data_id's that use that address.
    print('Uniquifying addresses to remove redundant conversions... ')
    addresses_df = addresses_df.groupby(['Address', 'City', 'State', 'ZIP Code'])['data_id'].apply(list).reset_index()

    # Duplicate the index of the address DataFrame into a new column to be used later during the geocoding.
    addresses_df = addresses_df.reset_index().rename(columns={'index': 'id'})
    addresses_df['id'] = addresses_df['id'].astype(str)

    return addresses_df, patient_data_df


def addresses_to_blockgroup(addresses_df, first_pass):
    """
    Converts addresses to U.S. Census Block Groups. The function is broken into 5 sub-functions.

    :param addresses_df: DataFrame that contains unique addresses that has the columns 'id', 'Address', 'City', 'State',
    'ZIP Code', and 'data_id'.
    :type addresses_df: pandas.DataFrame

    :param first_pass: Boolean that is used to direct whether the first-pass conversion using the
    addresses directly by the U.S. Census Geocoder should be done. The default value is True and is set to False with
    the --skip_first_pass flag.
    :type first_pass: bool

    :return: (DataFrame that contains successfully geocoded addresses, DataFrame that contains unsuccessfully geocoded
    addresses)
    :rtype: tuple
    """
    # 1. Filter out PO and Route boxes from address list as they cannot be meaningfully converted.
    no_box_addresses_df, po_and_route_box_df = filter_po_and_route_box_addresses(addresses_df)

    if first_pass:
        # 2A. Do a first-pass conversion of addresses directly using the U.S. Census' Geocoder if the skip-first-pass
        # flag is not used.
        first_pass_matched_df, unmatched_df = first_pass_geocode_using_addresses(no_box_addresses_df)
    else:
        # 2B. Set the first-pass matches DataFrame as empty and pass along the PO-and-Route-Box free address DataFrame
        # as first-pass unmatched DataFrame if the skip-first-pass flag is true.
        first_pass_matched_df = pandas.DataFrame(columns=no_box_addresses_df.columns)
        unmatched_df = no_box_addresses_df.copy()

    # 3. Geocode the addresses that failed the first-pass conversion into coordinates using Google and OpenStreetMaps.
    # Then separate the addresses that were successfully geocoded into coordinates from those that were not.
    coordinates_df, failed_addresses_coordinate_step_df = unmatched_addresses_to_coordinates(unmatched_df)

    # 4. Do a second-pass conversion of the addresses that were successfully converted into coordinates using the U.S.
    # Census Geocoder, this time passing in the coordinates instead of the addresses directly.
    second_pass_matched_df, failed_addresses_geocode_step_df = second_pass_geocode_using_coordinates(coordinates_df)

    # 5. Combine the addresses that were successfully geocoded, either using their addresses directly or through
    # coordinates. Also combine PO/Route Boxes addresses and addresses that failed to be geocoded, either into
    # coordinates or into U.S. Census Block Groups.
    converted_df, failed_df = \
        combine_first_and_second_pass_results(first_pass_matched_df, second_pass_matched_df, po_and_route_box_df,
                                              failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df)
    return converted_df, failed_df


def filter_po_and_route_box_addresses(addresses_df):
    """
    Filters out PO and Route boxes from the DataFrame of unique addresses.

    :param addresses_df: DataFrame that contains unique addresses that has the columns 'id', 'Address', 'City', 'State',
    'ZIP Code', and 'data_id'.
    :type addresses_df: pandas.DataFrame

    :return: (DataFrame of addresses that do not contain PO and Route boxes, DataFrame of addresses that do contain PO
    and Route boxes)
    :rtype: tuple
    """
    # The regular expression that identifies the majority of PO Boxes and Route Boxes.
    regex = r'[p|P][\s]*[o|O][\s]*[b|B][\s]*[o|O][\s]*[x|X][\s]*[a-zA-Z0-9]*' \
            r'|' \
            r'\b[P|p]+(?:OST|ost|o|O)?\.?\s*[O|o|0]+(?:ffice|FFICE)?\.?\s*[B|b][O|o|0]?[X|x]+\.?\s+[#]?(?:\d+)*(?:\D+)*\b' \
            r'|' \
            r'\b(?:R\.?R\.?|R\.?T\.?|ROUTE)\b[^,].+\b(?:BO?X)\b'

    # Identify PO and Route Boxes in the 'Address' column case insensitively.
    print('Filtering out PO Boxes and Route Boxes from addresses...')
    post_route_bool = addresses_df['Address'].str.contains(regex, regex=True, flags=re.IGNORECASE)

    # Filter out PO and Route Box addresses into a separate DataFrame.
    po_and_route_box_df = addresses_df.loc[post_route_bool]
    no_box_addresses_df = addresses_df.loc[~post_route_bool]

    return no_box_addresses_df, po_and_route_box_df


def first_pass_geocode_using_addresses(addresses_df):
    """
    Does the first-pass conversion of addresses to U.S. Census Block Groups using the addresses directly.

    :param addresses_df: DataFrame of filtered unique addresses that contains the necessary 4 columns: 'Address',
    'City', 'State', 'ZIP Code'.
    :type addresses_df: pandas.DataFrame

    :return: (DataFrame of addresses that were successfully geocoded in the first-pass, DataFrame of addresses that were
    not successfully geocoded in the first-pass)
    :rtype: tuple
    """
    # Return the empty DataFrame twice if the addresses DataFrame is empty, which would happen if all of the addresses
    # were PO/Route Boxes and were filtered out.
    if addresses_df.empty:
        return addresses_df, addresses_df

    # Create a list of DataFrames that are 100-row chunks of the complete set of addresses.
    number_of_rows_per_chunk = 100
    number_of_chunks = math.ceil(addresses_df.shape[0] / number_of_rows_per_chunk)

    print(f'Splitting address data into {number_of_chunks} {number_of_rows_per_chunk}-row chunks...')
    addresses_df_chunks = numpy.array_split(addresses_df, number_of_chunks)

    # Multithread the first-pass conversion of address chunks to Census Blocks Groups using the U.S. Census Geocoder.
    # This returns a list of successfully geocoded and unsuccessfully geocoded address DataFrame pairs as tuples.
    print('Running first-pass conversion of addresses to Census Block Groups using the U.S. Census Geocoder...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list_of_matched_and_unmatched_pairs = list(tqdm.tqdm(executor.map(batch_addresses_to_blockgroup,
                                                                          addresses_df_chunks),
                                                             total=len(addresses_df_chunks)))

    # Reorient the pair of tuples to a list that contains all the successfully converted address DataFrames and a list
    # that contains all the unsuccessfully converted address DataFrames.
    result = [list(i) for i in zip(*list_of_matched_and_unmatched_pairs)]

    # Concatenate the two lists of successful and unsuccessful address DataFrames into a large successful address
    # DataFrame and a large unsuccessful address DataFrame.
    matched_df, unmatched_df = pandas.concat(result[0], ignore_index=True), pandas.concat(result[1], ignore_index=True)

    # Combine the State, County, Tract, and Block ID/FIPS codes into the complete 15-digit FIPS code for each address.
    matched_df['FIPS'] = matched_df['statefp'] + matched_df['countyfp'] + matched_df['tract'] + matched_df['block']

    return matched_df, unmatched_df


def batch_addresses_to_blockgroup(unmatched_df):
    """
    Converts a batch of addresses to U.S. Census Block Groups using the censusgeocode package.

    :param unmatched_df: DataFrame chunk that contains subset of addresses from the main dataframe that contains
    filtered, unique addresses.
    :type unmatched_df: pandas.DataFrame

    :return: [DataFrame chunk of addresses that were successfully geocoded in the first-pass, DataFrame chunk of
    addresses that were not successfully geocoded in the first-pass]
    :rtype: list
    """
    # Save the column names of the DataFrame that contains unmatched addresses for later use.
    original_columns = unmatched_df.columns.values.tolist()

    # Create an empty list to be populated with DataFrames that contain successfully geocoded addresses.
    matched_df_list = []

    # Loop through the benchmark-vintage pairs.
    for benchmark, vintage in benchmark_vintage_tuple_list:

        # Convert the address columns of the DataFrame into a list of dictionaries where each dictionary is an address.
        address_dict_list = unmatched_df[['id', 'Address', 'City', 'State', 'ZIP Code']] \
            .rename(columns={'Address': 'street', 'City': 'city', 'State': 'state', 'ZIP Code': 'zip'}) \
            .to_dict('records')

        # Create an instance of the Censusgeocoder with the specified benchmark and vintage of the loop.
        cg = censusgeocode.CensusGeocode(benchmark=benchmark, vintage=vintage)

        # Run the batch of address through the U.S. Census Geocoder using the Cesusgeocode wrapper, returning a list of
        # dictionaries that each contain an address record.
        try:
            api_result_dict = cg.addressbatch(address_dict_list, timeout=2700)
        except:  # TODO: Identify the type of exceptions thrown to remove bare except catch.
            # If any error occurs, move onto the next iteration of the for loop.
            continue

        # Convert the list of address records in the form of dictionaries into a Pandas' DataFrame, which contains the
        # original id column, original address columns, FIPS code columns at the State, County, Tract, and Block level,
        # and coordinates columns.
        api_result_df = pandas.DataFrame(api_result_dict).drop(columns=['address', 'match', 'matchtype',
                                                                        'parsed', 'tigerlineid', 'side'])

        # Join the converted address columns back to the unmatched DataFrame using the 'id' column and replace empty
        # whitespace with NaN's.
        merged_api_df = unmatched_df.merge(api_result_df, on='id')
        merged_api_df.replace(r'^\s*$', numpy.NaN, regex=True, inplace=True)

        # To the list of matched address DataFrames, append a DataFrame that is subset of the returned DataFrame where
        # a value was found for the U.S. Census Block.
        matched_df_list.append(merged_api_df[merged_api_df['block'].notna()])

        # Set the unmatched address DataFrame to be the addresses that did not have a value for the U.S. Census Block
        # in the returned DataFrame.
        unmatched_df = merged_api_df[merged_api_df['block'].isna()][original_columns]

        # If the new unmatched address DataFrame is empty, exit the for-loop as that all addresses in this chunk have
        # been successfully geocoded; else, continue to the next loop using the new unmatched address DataFrame.
        if unmatched_df.empty:
            break

    # Combine the list of successfully geocoded address DataFrames into one, large DataFrame.
    matched_df = pandas.concat(matched_df_list, ignore_index=True)

    return [matched_df, unmatched_df]


def unmatched_addresses_to_coordinates(unmatched_df):
    """
    Converts addresses that were not successfully geocoded in the first-pass conversion of lat-lon coordinates.

    :param unmatched_df: DataFrame of addresses that were not successfully geocoded in the first-pass conversion using
    the U.S. Census Geocoder directly.
    :type unmatched_df: pandas.DataFrame

    :return: (DataFrame of addresses that were successfully geocoded into coordinates, DataFrame of addresses that were
    not successfully geocoded into coordinates)
    :rtype: tuple
    """
    # Return an empty DataFrame twice if the unmatched address DataFrame is empty, which would happen if all of the
    # addresses were successfully converted in the first-pass.
    if unmatched_df.empty:
        return unmatched_df, unmatched_df

    print('Geocoding addresses that failed in the first-pass conversion into coordinates using Google...')

    # Create a tqdm progress bar for Pandas' apply function that will be used.
    tqdm.tqdm.pandas(position=0, leave=True)

    # Geocode each address that failed to geocode in the first-pass conversion to coordinates using Google and set the
    # result as 'lat' and 'lon' in the unmatched DataFrame.
    unmatched_df[['lat', 'lon']] = unmatched_df.progress_apply(address_to_coordinates_with_google,
                                                               axis=1, result_type='expand')

    # Separate the addresses that were successfully geocoded into coordinates using Google from those that were not.
    google_coordinates_df = unmatched_df[unmatched_df['lat'].notna()]
    google_failed_addresses_coordinate_step_df = unmatched_df[unmatched_df['lat'].isna()].drop(columns=['lat', 'lon'])

    # If all of the unmatched addresses were successfully converted into coordinates using Google, return the DataFrame
    # of Google-geocoded address coordinates and the empty DataFrame of addresses that failed to be geocoded by Google.
    if google_failed_addresses_coordinate_step_df.empty:
        return google_coordinates_df, google_failed_addresses_coordinate_step_df

    print('Geocoding addresses that failed to be geocoded by Google into coordinates using Nominatim/OpenStreetMaps...')

    # Instantiating a Nominatim locator used for geocoding and rate-limiting it.
    locator = geopy.geocoders.Nominatim(user_agent="my_application")
    geocode = geopy.extra.rate_limiter.RateLimiter(locator.geocode, min_delay_seconds=0.5, error_wait_seconds=60)

    # Geocode each address failed by Google using Nominatim/OpenStreetMaps and store the coordinate results.
    google_failed_addresses_coordinate_step_df[['lon', 'lat']] = google_failed_addresses_coordinate_step_df \
        .progress_apply(lambda row: address_to_coordinates_with_nominatim(row, geocode), axis=1, result_type='expand')

    # Separate address that were successfully geocoded into coordinates using Nominatim/OpenStreetMaps from those that
    # were not.
    nominatim_coordinates_df = google_failed_addresses_coordinate_step_df[
        google_failed_addresses_coordinate_step_df['lon'].notna()]
    failed_addresses_coordinate_step_df = google_failed_addresses_coordinate_step_df[
        google_failed_addresses_coordinate_step_df['lon'].isna()]

    # Combine the addresses that were successfully converted to addresses using either Google or
    # Nominatim/OpenStreetMaps into a DataFrame.
    coordinates_df = pandas.concat([google_coordinates_df, nominatim_coordinates_df], ignore_index=True)

    return coordinates_df, failed_addresses_coordinate_step_df


def address_to_coordinates_with_google(row):
    """
    Geocode address to coordinates with Google.

    :param row: A row from the DataFrame that contains addresses that were not successfully geocoded in the first-pass.
    :type row: pandas.Series

    :return: (Longitude, Latitude) or (None, None)
    :rtype: tuple
    """
    # Set variables for the base Google search URL, headers, and query parameters, which include a string that is
    # concat of the address information.
    base_url = 'https://www.google.com/search'
    headers = {'User-Agent': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'}
    params = {'source': 'opera', 'q': " ".join([str(row['Address']), str(row['City']),
                                                str(row['State']), str(row['ZIP Code'])])}
    try:
        # Run a GET requests using the base URL, headers, and parameters set above.
        base_google_search_request = requests.get(base_url, headers=headers, params=params)

        # If a HTTP 200 response code is received, search for URLs in the HTML body that begin with the URL snippet
        # below to identify Google Maps URL that may have coordinate information.
        if base_google_search_request.status_code == 200:
            google_map_urls = re.findall(r'https://maps.google.com/maps\?um=1\S+(?=")', base_google_search_request.text)

            # If a URL/URLs is/are found, loop through them.
            if google_map_urls:
                for google_map_url in google_map_urls:

                    # Search to see if the URL contains coordinates. If so, convert the coordinates into a tuple of
                    # floats and return the tuple. Else, make a GET request with the URL.
                    coords = re.search(r'(?<=ll=)-?\d{1,2}(?:\d+|.\d+),-?\d{1,3}(?:\d+|.\d+)', google_map_url)
                    if coords:
                        coords_string = coords.group()
                        return eval(f'({coords_string})')
                    else:
                        google_directions_request = requests.get(google_map_url)

                        # If a HTTP 200 response code is received, search the HTML body for the URL snippet below to
                        # pull coordinates from it.
                        if google_directions_request.status_code == 200:
                            google_static_image_urls = re.findall(
                                r'https://maps.google.com/maps/api/staticmap\?\S+(?=")', google_directions_request.text)

                            # If the URL(s) is/are found, loop through them and search for coordinates in the URL.
                            if google_static_image_urls:
                                for google_static_image_url in google_static_image_urls:
                                    coords = re.search(
                                        r'(?<=markers=)-?\d{1,2}(?:\d+|.\d+)%2C-?\d{1,2}(?:\d+|.\d+)(?=&)',
                                        google_static_image_url)

                                    # If coordinates are found, convert them into a tuple of floats and return the
                                    # tuple.
                                    if coords:
                                        coords_string = coords.group()
                                        coords_pair_list = [float(i) for i in coords_string.split('%2C')]
                                        return tuple(coords_pair_list)
        # If everything is looped and branched through and no coordinates are found, return None, None as placeholders
        # for missing coordinates.
        return None, None
    except:  # TODO: Identify the type of exceptions thrown to remove bare except catch.
        # If any error occurs, return None, None as placeholders for missing coordinates.
        return None, None


def address_to_coordinates_with_nominatim(row, geocode):
    """
    Geocode address to coordinates using Nominatim/OpenStreetMaps

    :param row: A row from the DataFrame that contains addresses that were not successfully geocoded in the first-pass.
    :type row: pandas.Series

    :param geocode: RateLimited Locator that is used to make API calls to Nominatim/OpenStreetMaps.
    :type geocode: geopy.extra.rate_limiter.RateLimiter

    :return: (Longitude, Latitude) or (None, None)
    :rtype: tuple
    """
    # Query Nominatim/OpenStreetMaps with the Address, State, and ZIP Code. If a location is found, return its
    # coordinates. Else, return None, None as placeholders for missing coordinates.
    try:
        location = geocode(query={'street': row['Address'], 'state': row['State'],
                                  'country': 'United States', 'postalcode': row['ZIP Code'][:5]},
                           country_codes='us')
        if location:
            lon, lat = location.longitude, location.latitude
        else:
            lon, lat = None, None
        return lon, lat
    except:  # TODO: Identify the type of exceptions thrown to remove bare except catch.
        # If any error occurs, return None, None as placeholders for missing coordinates.
        return None, None


def second_pass_geocode_using_coordinates(coordinates_df):
    """
    Does the second-pass conversion of addresses in the form of coordinates to U.S. Census Block Groups using the U.S.
    Census Geocoder.

    :param coordinates_df: DataFrame containing coordinates of addresses that were successfully geocoded using Google or
    Nominatim/OpenStreetMaps.
    :type coordinates_df: pandas.DataFrame

    :return: (DataFrame containing addresses that were successfully geocoded during the second-pass conversion using
    coordinates, DataFrame containing addresses that were not successfully geocoded during the second-pass conversion
    using coordinates)
    :rtype: tuple
    """
    # Return the empty DataFrame twice if the newly geocoded coordinates DataFrame is empty, which would happen if none
    # of the addresses that failed to be geocoded in the first-pass were successfully geocoded to coordinates.
    if coordinates_df.empty:
        return coordinates_df, coordinates_df

    print('Running second-pass conversion of addresses, this time as coordinates, to Census Block Groups using the '
          'U.S. Census Geocoder...')

    # Multithread the second-pass conversion of addresses represented by coordinates to U.S. Census Block Groups using
    # the U.S. Census Geocoder. Set the returned list of FIPS codes as the 'FIPS' column in the coordinates DataFrame.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        coordinates_df['FIPS'] = list(tqdm.tqdm(executor.map(coordinates_to_blockgroup,
                                                             coordinates_df['lon'], coordinates_df['lat']),
                                                total=coordinates_df.shape[0]))

    # Set addresses whose coordinates were not found by the U.S. Census Geocoder to a failed address DataFrame.
    failed_addresses_geocode_step_df = coordinates_df[coordinates_df['FIPS'].isna()].drop(columns='FIPS')

    # Set the addresses whose coordinates were found by the U.S. Census Geocoder to a matched DataFrame. Splice the
    # string in the 'FIPS' column into State, County, Tract, and Block ID/FIPS codes to match DataFrame of successfully
    # matched addresses from the first-pass conversion.
    matched_df = coordinates_df.dropna(subset=['FIPS'])
    matched_df['statefp'] = matched_df['FIPS'].str.slice(stop=2)
    matched_df['countyfp'] = matched_df['FIPS'].str.slice(start=3, stop=5)
    matched_df['tract'] = matched_df['FIPS'].str.slice(start=6, stop=11)
    matched_df['block'] = matched_df['FIPS'].str.slice(start=12)

    return matched_df, failed_addresses_geocode_step_df


def coordinates_to_blockgroup(longitude, latitude):
    """
    Geocodes a coordinate to a U.S. Census Block Group using the U.S. Census Geocoder.

    :param longitude: Longitude or 'X' coordinate.
    :type longitude: float

    :param latitude: Latitude or 'Y' coordinate.
    :type latitude: float

    :return: 15-digit FIPS code or None
    :rtype: str or None
    """
    # Loop through the benchmark-vintage pairs and instantiate a Censusgeocoder with specified benchmark and vintages.
    for benchmark, vintage in benchmark_vintage_tuple_list:
        cg = censusgeocode.CensusGeocode(benchmark=benchmark, vintage=vintage)

        # Get the response and if in the result there was match whose GEOID string length was 15, which would correspond
        # to the U.S. Census Block Group FIPS code, return that value. If a GEOID whose length is 15 wsa not found or an
        # error occurred, jump to the next iteration of the for-loop.
        try:
            result = cg.coordinates(x=longitude, y=latitude)
        except:  # TODO: Identify the type of exceptions thrown to remove bare except catch.
            continue
        if result:
            for match in result.keys():
                if 'GEOID' in result[match][0]:
                    if len(result[match][0]['GEOID']) == 15:
                        return result[match][0]['GEOID']

    # If a GEOID whose length is 15 was not found through all of the loops, return None as a placeholder for the missing
    # FIPS code.
    return None


def combine_first_and_second_pass_results(first_pass_matched_df, second_pass_matched_df, po_and_route_box_df,
                                          failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df):
    """
    Combine addresses that were successfully geocoded to U.S. Census Block Groups together into a dataframe and combine
    PO/Route box addresses, addresses that failed to be geocoded to coordinates, and addresses that failed to be
    geocoded to U.S. Census Block Groups as coordinates into a dataframe.

    :param first_pass_matched_df: DataFrame of addresses that were successfully geocoded to U.S. Census Block Groups in
    the first-pass.
    :type first_pass_matched_df: pandas.DataFrame

    :param second_pass_matched_df: DataFrame of addresses that were successfully geocoded to U.S. Census Block Groups in
    the second-pass
    :type first_pass_matched_df: pandas.DataFrame

    :param po_and_route_box_df: DataFrame of addresses that are PO and Route boxes.
    :type first_pass_matched_df: pandas.DataFrame

    :param failed_addresses_coordinate_step_df: DataFrame of addresses that failed to be geocoded into coordinates.
    :type first_pass_matched_df: pandas.DataFrame

    :param failed_addresses_geocode_step_df: DataFrame of addresses that failed to be geocoded to U.S. Census Block
    Groups using coordinates in the second-pass.
    :type first_pass_matched_df: pandas.DataFrame

    :return: (DataFrame of addresses that were successfully geocoded and have their associated U.S. Census Block Groups,
    DataFrame of addresses that were not successfully geocoded into U.S. Census Block Groups)
    :rtype: tuple
    """
    print('Combining successfully geocode addresses and unsuccessfully geocoded addresses into two large DataFrames...')

    # Concatenate addresses that were successfully geocoded in either the first-pass or second-pass together.
    addresses_blockgroup_df = pandas.concat([first_pass_matched_df, second_pass_matched_df], ignore_index=True)

    # Concatenate addresses that failed to be geocoded due the address being a PO/Route box, being unable to be geocoded
    # to a coordinate, or being able to converted from coordinate to U.S. Census Block.
    failed_df = pandas.concat([po_and_route_box_df, failed_addresses_coordinate_step_df,
                               failed_addresses_geocode_step_df], ignore_index=True)

    return addresses_blockgroup_df, failed_df


def blockgroup_to_adi(addresses_blockgroup_df):
    """
    Maps Area Deprivation Indices to successfully geocoded addresses through U.S. Census Block Groups.

    :param addresses_blockgroup_df: DataFrame of addresses that were successfully geocoded and have their associated
    U.S. Census Block Groups
    :param addresses_blockgroup_df: pandas.DataFrame

    :return: DataFrame of addresses that were successfully geocoded and have their associated U.S. Census Block Groups
    and Area Deprivation Indices.
    :rtype: pandas.DataFrame
    """
    print('Mapping Area Deprivation Indices to successfully geocoded addresses using U.S. Census Block Groups...')

    # Import the Block Group to ADI DataFrame from the BlockGroupToADI.feather file.
    with importlib.resources.path('geocodeadi.resources', 'BlockGroupToADI.feather') as blockgroup_to_adi_feather_path:
        blockgroup_to_adi_df = pandas.read_feather(blockgroup_to_adi_feather_path)

    # Slice FIPS code column to the first 12-digits and use it as a key to map Area Deprivation Indices to successfully
    # geocoded addresses.
    addresses_blockgroup_df['FIPS'] = addresses_blockgroup_df['FIPS'].str.slice(stop=12)
    addresses_blockgroup_adi_df = addresses_blockgroup_df.merge(blockgroup_to_adi_df, on='FIPS')

    return addresses_blockgroup_adi_df


def export_data(successful_df, failed_df, patient_data_df):
    """
    Exports dataframe of successfully geocoded addresses and associated U.S. Census Block Groups and Area Deprivation
    Indices as well as dataframe of addresses that were not successfully geocoded.

    :param successful_df: DataFrame of addresses that were successfully geocoded and have their associated U.S. Census
    Block Groups and Area Deprivation Indices.
    :type successful_df: pandas.DataFrame

    :param failed_df: DataFrame of addresses that were not successfully geocoded into U.S. Census Block Groups.
    :type failed_df: pandas.DataFrame

    :param patient_data_df: DataFrame that extra patient information that was provided in the original data.
    :type patient_data_df: pandas.DataFrame
    """
    print(
        'Exporting successfully and unsuccessfully geocoded address DataFrames to successful.csv and unsuccessful.csv '
        'respectively...')

    # Explode the values in the 'data_id' column for the successful and failed address DataFrame, so that each row has
    # its own data_id and where it is possible for multiple data_ids to share the same address. Drop the 'id' column
    # that was used to identify unique addresses.
    successful_df = successful_df.explode('data_id', ignore_index=True).drop(columns='id')
    failed_df = failed_df.explode('data_id', ignore_index=True).drop(columns='id')

    # Merge the address data back to the patient data to recreate the original data that was provided, now with geocode
    # and ADI information.
    successful_df = patient_data_df.merge(successful_df, on='data_id').drop(columns='data_id')
    failed_df = patient_data_df.merge(failed_df, on='data_id').drop(columns='data_id')

    # Export the successfully and unsuccessfully geocoded DataFrames to CSV.
    successful_df.to_csv('successful.csv', index=False)
    failed_df.to_csv('failed.csv', index=False)
