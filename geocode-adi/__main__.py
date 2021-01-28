import concurrent.futures
import csv
import importlib.resources
import math
import os
import sys
import warnings

import censusgeocode
import geopy
import geopy.exc
import geopy.extra.rate_limiter
import numpy
import pandas
import tqdm

benchmark_vintage_pairs_dict = {
    'Public_AR_Current': ['Census2010_Current', 'ACS2019_Current', 'ACS2018_Current', 'ACS2017_Current'],
    'Public_AR_TAB2020': ['ACS2019_TAB2020', 'ACS2018_TAB2020', 'ACS2017_TAB2020', 'Census2010_TAB2020'],
    'Public_AR_Census2010': ['Census2010_Census2010', 'Census2000_Census2010']}
benchmark_vintage_tuple_list = [(benchmark, vintage)
                                for benchmark, vintages in benchmark_vintage_pairs_dict.items()
                                for vintage in vintages]

warnings.simplefilter('ignore', FutureWarning)
pandas.options.mode.chained_assignment = None


def main():
    """
    Main function that contains subfunctions that convert addresses into ADI values based on Census Block Groups.
    The subfunctions line up with the 5 overarching steps:
    """

    # 1. Check datafile arguments.
    check_datafile_arg()

    # 2. Read in addresses from datafile.
    addresses_df, patient_data_df = read_in_data()

    # 3. Geocode addresses to Census Block Groups.
    addresses_blockgroup_df, failed_addresses_df = addresses_to_blockgroup(addresses_df)

    # 4. Add associated ADI value for each Census Block Group.
    addresses_blockgroup_adi_df = blockgroup_to_adi(addresses_blockgroup_df)

    # 5. Export new dataframe that contains addresses, Census Block Group, and ADI information to CSV as well as
    # failed addresses.
    export_data(addresses_blockgroup_adi_df, failed_addresses_df, patient_data_df)


def check_datafile_arg():
    """
    Function that checks that only one argument was provided and that the path to the datafile exists.
    """

    print('Checking datafile argument...')

    # Check to see more/less than 1 argument was given.
    if len(sys.argv) != 2:
        if len(sys.argv) < 2:
            raise ValueError('Missing datafile argument. Run again with proper comma-delimited file of addresses.')
        else:
            raise ValueError('Extra arguments given. Run again with the only argument being the path to the datafile.')

    # Check to see if the datafile path does not exist.
    if not os.path.exists(sys.argv[1]):
        raise ValueError('Provided datafile path does not exist.')

    # Arguments check out. Continue on with the code.
    return


def read_in_data():
    """
    Function that reads command-delimited datafile that contains at least the columns titled: 'Address', 'City',
    'State', 'ZIP Code'.

    :return:
    """

    print('Reading in addresses...')
    df = pandas.read_csv(sys.argv[1], low_memory=False, dtype=str) \
        .reset_index().rename(columns={'index': 'data_id'})

    print('Separating addresses data from other patient data...')
    addresses_df = df[['data_id', 'Address', 'City', 'State', 'ZIP Code']]
    patient_data_df = df.drop(columns=['Address', 'City', 'State', 'ZIP Code'])

    print('Uniquefying addresses and creating a key back to original data...')
    addresses_df = addresses_df.groupby(['Address', 'City', 'State', 'ZIP Code'])['data_id'].apply(list).reset_index()
    addresses_df = addresses_df.reset_index().rename(columns={'index': 'id'})
    addresses_df['id'] = addresses_df['id'].astype(str)

    return addresses_df, patient_data_df


def addresses_to_blockgroup(addresses_df):
    """
    Main addresses to block group conversion function which is broken up into 4 subfunctions for the 4 overarching
    steps.

    :param addresses_df:
    :return:
    """

    # 1. Do a first-pass conversion of address to Census Block Groups by the U.S. Census Geocoder using the address,
    # separating results into two groups: first-pass matched addresses and unmatched addresses.
    first_pass_matched_df, unmatched_df = first_pass_geocode_using_addresses(addresses_df)

    # 2. Try and convert the unmatched addresses to latitude-longitude coordinates using Nominatim's OpenStreetMaps API,
    # returning addresses that were successfully converted to lat-long coordinates and those that were not.
    coordinates_df, failed_addresses_coordinate_step_df = unmatched_addresses_to_coordinates(unmatched_df)

    # 3. Do a second-pass conversion of address to Census Block Groups by the lat-long coordinates, separating results
    # into two groups: second-pass matched addresses and failed addresses.
    second_pass_matched_df, failed_addresses_geocode_step_df = second_pass_geocode_using_coordinates(coordinates_df)

    # 4. Combine the successfully matched addresses from the first and second passes together and the failed
    # addresses from the coordinate and address geocode step.
    converted_df, failed_df = \
        combine_first_and_second_pass_result(first_pass_matched_df, second_pass_matched_df,
                                             failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df)
    return converted_df, failed_df


def first_pass_geocode_using_addresses(addresses_df):
    """
    Function that does the first-pass conversion of addresses to Census Block Groups using the U.S. Census Geocoder.
    :param addresses_df:
    :return:
    """

    # Split the complete dataframe of addresses into <= 10,000 row chunks for the U.S. Census Geocoder batch conversion.

    number_of_rows_per_chunk = 50
    number_of_chunks = math.ceil(addresses_df.shape[0] / number_of_rows_per_chunk)

    print(f'Splitting addresses dataframe into {number_of_chunks} chunks of {number_of_rows_per_chunk} addresses...')
    addresses_df_chunks = numpy.array_split(addresses_df, number_of_chunks)

    # Use a concurrent.futures map to multithread address to Census Block Group geocoding batch requests. Use tqdm to
    # visual progress bar. Output a list of lists, where each nested list contains the pair of matched and unmatched
    # address dataframe chunks outputted from batch_addresses_to_blockgroup function.
    print('Running first-pass conversion of addresses to Census Block Groups using U.S. Census Geocoder...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list_of_matched_and_unmatched_pairs = list(tqdm.tqdm(executor.map(batch_addresses_to_blockgroup,
                                                                          addresses_df_chunks),
                                                             total=len(addresses_df_chunks)))

    # Separate the list of lists that contain matched-unmatched dataframe pairs to a larger list that contains two
    # list: one list for all of the matched dataframes and another for all of the unmatched dataframes.
    result = [list(i) for i in zip(*list_of_matched_and_unmatched_pairs)]

    # Concatenate the two individual lists of matched and unmatched dataframes to two large dataframes.
    print('Combining first pass results into matched and unmatched dataframes...')
    matched_df, unmatched_df = pandas.concat(result[0], ignore_index=True), pandas.concat(result[1], ignore_index=True)
    matched_df['FIPS'] = matched_df['statefp'] + matched_df['countyfp'] + matched_df['tract'] + matched_df['block']

    return matched_df, unmatched_df


def batch_addresses_to_blockgroup(unmatched_df):
    original_columns = unmatched_df.columns.values.tolist()
    matched_df_list = []
    for benchmark, vintage in benchmark_vintage_tuple_list:
        address_dict_list = unmatched_df[['id', 'Address', 'City', 'State', 'ZIP Code']] \
            .rename(columns={'Address': 'street', 'City': 'city', 'State': 'state', 'ZIP Code': 'zip'}) \
            .to_dict('records')
        cg = censusgeocode.CensusGeocode(benchmark=benchmark, vintage=vintage)
        try:
            api_result_dict = cg.addressbatch(address_dict_list, timeout=2700)
        except csv.Error:
            continue
        api_result_df = pandas.DataFrame(api_result_dict).drop(columns=['address', 'match', 'matchtype',
                                                                        'parsed', 'tigerlineid', 'side'])
        merged_api_df = unmatched_df.merge(api_result_df, on='id')
        merged_api_df.replace(r'^\s*$', numpy.NaN, regex=True, inplace=True)
        matched_df_list.append(merged_api_df[merged_api_df['block'].notna()])
        unmatched_df = merged_api_df[merged_api_df['block'].isna()][original_columns]
        if unmatched_df.empty:
            break

    matched_df = pandas.concat(matched_df_list)
    return [matched_df, unmatched_df]


def unmatched_addresses_to_coordinates(unmatched_df):
    print('Converting first-pass unmatched addresses to latitude/longitude coordinates using Nominatim API...')
    tqdm.tqdm.pandas(position=0, leave=True)
    unmatched_df[['lon', 'lat']] = unmatched_df.progress_apply(address_to_coordinates, axis=1, result_type='expand')

    print('Combining unmatched addresses to coordinates results into successful and failed dataframes...')
    coordinates_df = unmatched_df[unmatched_df['lon'].notna()]
    failed_addresses_coordinate_step_df = unmatched_df[unmatched_df['lon'].isna()]
    return coordinates_df, failed_addresses_coordinate_step_df


def address_to_coordinates(row):
    try:
        locator = geopy.geocoders.Nominatim(user_agent="my_application")
        geocode = geopy.extra.rate_limiter.RateLimiter(locator.geocode, min_delay_seconds=1, error_wait_seconds=60)
        location = geocode(query={'street': row['Address'], 'state': row['State'],
                                  'country': 'United States', 'postalcode': row['ZIP Code'][:5]},
                           country_codes='us')
    except geopy.exc.GeocoderUnavailable:
        return None, None
    if location:
        lon, lat = location.longitude, location.latitude
    else:
        lon, lat = None, None
    return lon, lat


def second_pass_geocode_using_coordinates(coordinates_df):
    print('Running second-pass conversion of addresses coordinates to Census Block Groups using '
          'U.S. Census Geocoder...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        coordinates_df['FIPS'] = list(tqdm.tqdm(executor.map(coordinates_to_blockgroup,
                                                             coordinates_df['lon'], coordinates_df['lat']),
                                                total=coordinates_df.shape[0]))

    print('Combining second-pass results into matched and unmatched dataframes...')
    failed_addresses_geocode_step_df = coordinates_df[coordinates_df['FIPS'].isna()].drop(columns='FIPS')

    matched_df = coordinates_df.dropna(subset=['FIPS'])
    matched_df['statefp'] = matched_df['FIPS'].str.slice(stop=2)
    matched_df['countyfp'] = matched_df['FIPS'].str.slice(start=3, stop=5)
    matched_df['tract'] = matched_df['FIPS'].str.slice(start=6, stop=11)
    matched_df['block'] = matched_df['FIPS'].str.slice(start=12)

    return matched_df, failed_addresses_geocode_step_df


def coordinates_to_blockgroup(longitude, latitude):
    for benchmark, vintage in benchmark_vintage_tuple_list:
        cg = censusgeocode.CensusGeocode(benchmark=benchmark, vintage=vintage)
        try:
            result = cg.coordinates(x=longitude, y=latitude)
        except ValueError:
            continue
        if result:
            for match in result.keys():
                if 'GEOID' in result[match][0]:
                    if len(result[match][0]['GEOID']) == 15:
                        return result[match][0]['GEOID']
    return None


def combine_first_and_second_pass_result(first_pass_matched_df, second_pass_matched_df,
                                         failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df):
    print('Combining successful matches from first and second passes together as well as combining failed addresses '
          'together...')
    addresses_blockgroup_df = pandas.concat([first_pass_matched_df, second_pass_matched_df], ignore_index=True)
    failed_df = pandas.concat([failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df],
                              ignore_index=True)
    return addresses_blockgroup_df, failed_df


def blockgroup_to_adi(addresses_blockgroup_df):
    print('Adding national and state ADI ranks to the dataframe of successful matches based on Census Block Group '
          'codes...')
    with importlib.resources.path('geocode-adi.resources', 'BlockGroupToADI.feather') as blockgroup_to_adi_feather_path:
        blockgroup_to_adi_df = pandas.read_feather(blockgroup_to_adi_feather_path)
    addresses_blockgroup_df['FIPS'] = addresses_blockgroup_df['FIPS'].str.slice(stop=12)
    addresses_blockgroup_adi_df = addresses_blockgroup_df.merge(blockgroup_to_adi_df, on='FIPS')
    return addresses_blockgroup_adi_df


def export_data(successful_df, failed_df, patient_data_df):
    print('Combining successful and failed addresses back with their original data...')
    successful_df = successful_df.explode('data_id', ignore_index=True).drop(columns='id')
    failed_df = failed_df.explode('data_id', ignore_index=True).drop(columns='id')

    successful_df = patient_data_df.merge(successful_df, on='data_id')
    failed_df = patient_data_df.merge(failed_df, on='data_id')

    print('Exporting successful and failed conversion data to CSV...')
    successful_df.to_csv('successful.csv', index=False)
    failed_df.to_csv('failed.csv', index=False)


main()
