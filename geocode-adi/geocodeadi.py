import concurrent.futures
import csv
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

benchmark_vintage_pairs_dict = {
    'Public_AR_Current': ['Census2010_Current', 'ACS2019_Current', 'ACS2018_Current', 'ACS2017_Current'],
    'Public_AR_TAB2020': ['ACS2019_TAB2020', 'ACS2018_TAB2020', 'ACS2017_TAB2020', 'Census2010_TAB2020'],
    'Public_AR_Census2020': ['Census2010_Census2020']}
benchmark_vintage_tuple_list = [(benchmark, vintage)
                                for benchmark, vintages in benchmark_vintage_pairs_dict.items()
                                for vintage in vintages]

warnings.simplefilter('ignore', FutureWarning)
pandas.options.mode.chained_assignment = None


def addresses_to_adi(address_path, first_pass):
    # 1. Read in addresses from datafile.
    addresses_df, patient_data_df = read_in_data(address_path)

    addresses_df.to_csv('addresses_df.csv')
    patient_data_df.to_csv('patient_data_df.csv')

    # 2. Geocode addresses to Census Block Groups.
    addresses_blockgroup_df, failed_addresses_df = addresses_to_blockgroup(addresses_df, first_pass)

    addresses_blockgroup_df.to_csv('addresses_blockgroup_df.csv')
    failed_addresses_df.to_csv('failed_addresses_df.csv')

    # 3. Add associated ADI value for each Census Block Group.
    addresses_blockgroup_adi_df = blockgroup_to_adi(addresses_blockgroup_df)

    addresses_blockgroup_adi_df.to_csv('addresses_blockgroup_adi_df.csv')

    # 4. Export new dataframe that contains addresses, Census Block Group, and ADI information to CSV as well as
    # failed addresses.
    export_data(addresses_blockgroup_adi_df, failed_addresses_df, patient_data_df)


### Step 1 ###

def read_in_data(address_path):
    print('Reading in addresses...')
    df = pandas.read_csv(address_path, low_memory=False, dtype=str) \
        .reset_index().rename(columns={'index': 'data_id'})

    print('Separating addresses data from other patient data...')
    addresses_df = df[['data_id', 'Address', 'City', 'State', 'ZIP Code']]
    patient_data_df = df.drop(columns=['Address', 'City', 'State', 'ZIP Code'])

    print('Uniquefying addresses and creating a key back to original data...')
    addresses_df = addresses_df.groupby(['Address', 'City', 'State', 'ZIP Code'])['data_id'].apply(list).reset_index()
    addresses_df = addresses_df.reset_index().rename(columns={'index': 'id'})
    addresses_df['id'] = addresses_df['id'].astype(str)

    return addresses_df, patient_data_df


### Step 2 ###

def addresses_to_blockgroup(addresses_df, first_pass):
    # 1.
    no_box_addresses_df, po_and_route_box_df = filter_po_and_route_box_addresses(addresses_df)

    no_box_addresses_df.to_csv('no_box_addresses_df.csv')
    po_and_route_box_df.to_csv('po_and_route_box_df.csv')

    if first_pass:
        # 2.
        first_pass_matched_df, unmatched_df = first_pass_geocode_using_addresses(no_box_addresses_df)
    else:
        first_pass_matched_df = pandas.DataFrame(columns=no_box_addresses_df.columns)
        unmatched_df = no_box_addresses_df.copy()

    first_pass_matched_df.to_csv('first_pass_matched_df.csv')
    unmatched_df.to_csv('unmatched_df.csv')

    # 3.
    coordinates_df, failed_addresses_coordinate_step_df = unmatched_addresses_to_coordinates(unmatched_df)

    coordinates_df.to_csv('coordinates_df.csv')
    failed_addresses_coordinate_step_df.to_csv('failed_addresses_coordinate_step_df.csv')

    # 4.
    second_pass_matched_df, failed_addresses_geocode_step_df = second_pass_geocode_using_coordinates(coordinates_df)

    second_pass_matched_df.to_csv('second_pass_matched_df.csv')
    failed_addresses_geocode_step_df.to_csv('failed_addresses_geocode_step_df.csv')

    # 5.
    converted_df, failed_df = \
        combine_first_and_second_pass_results(first_pass_matched_df, second_pass_matched_df, po_and_route_box_df,
                                              failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df)
    return converted_df, failed_df


### Step 2.1 ###
def filter_po_and_route_box_addresses(addresses_df):
    print("Removing Post and Route Boxes from address list...")

    regex = r'[p|P][\s]*[o|O][\s]*[b|B][\s]*[o|O][\s]*[x|X][\s]*[a-zA-Z0-9]*' \
            r'|' \
            r'\b[P|p]+(?:OST|ost|o|O)?\.?\s*[O|o|0]+(?:ffice|FFICE)?\.?\s*[B|b][O|o|0]?[X|x]+\.?\s+[#]?(?:\d+)*(?:\D+)*\b' \
            r'|' \
            r'\b(?:R\.?R\.?|R\.?T\.?|ROUTE)\b[^,].+\b(?:BO?X)\b'
    post_route_bool = addresses_df['Address'].str.contains(regex, regex=True, flags=re.IGNORECASE)

    po_and_route_box_df = addresses_df.loc[post_route_bool]
    no_box_addresses_df = addresses_df.loc[~post_route_bool]

    return no_box_addresses_df, po_and_route_box_df


### Step 2.2 ###
def first_pass_geocode_using_addresses(addresses_df):
    if addresses_df.empty:
        return addresses_df, addresses_df

    number_of_rows_per_chunk = 100
    number_of_chunks = math.ceil(addresses_df.shape[0] / number_of_rows_per_chunk)

    print(f'Splitting addresses dataframe into {number_of_chunks} chunks of {number_of_rows_per_chunk} addresses...')
    addresses_df_chunks = numpy.array_split(addresses_df, number_of_chunks)

    print('Running first-pass conversion of addresses to Census Block Groups using U.S. Census Geocoder...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list_of_matched_and_unmatched_pairs = list(tqdm.tqdm(executor.map(batch_addresses_to_blockgroup,
                                                                          addresses_df_chunks),
                                                             total=len(addresses_df_chunks)))

    result = [list(i) for i in zip(*list_of_matched_and_unmatched_pairs)]

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


### Step 2.3 ###
def unmatched_addresses_to_coordinates(unmatched_df):
    if unmatched_df.empty:
        return unmatched_df, unmatched_df

    tqdm.tqdm.pandas(position=0, leave=True)

    print('Converting first-pass unmatched addresses to latitude/longitude coordinates using Google')

    unmatched_df[['lat', 'lon']] = unmatched_df.progress_apply(
        address_to_coordinates_with_google, axis=1, result_type='expand')

    google_coordinates_df = unmatched_df[unmatched_df['lat'].notna()]
    google_failed_addresses_coordinate_step_df = unmatched_df[unmatched_df['lat'].isna()].drop(columns=['lat', 'lon'])

    google_coordinates_df.to_csv('google_coordinates_df.csv')
    google_failed_addresses_coordinate_step_df.to_csv('google_failed_addresses_coordinate_step_df.csv')

    if google_failed_addresses_coordinate_step_df.empty:
        return google_coordinates_df, google_failed_addresses_coordinate_step_df

    print('Converting first-pass unmatched addresses to latitude/longitude coordinates using Nominatim...')

    locator = geopy.geocoders.Nominatim(user_agent="my_application")
    geocode = geopy.extra.rate_limiter.RateLimiter(locator.geocode, min_delay_seconds=0.5, error_wait_seconds=60)

    google_failed_addresses_coordinate_step_df[['lon', 'lat']] = google_failed_addresses_coordinate_step_df \
        .progress_apply(lambda row: address_to_coordinates_with_nominatim(row, geocode), axis=1, result_type='expand')

    nominatim_coordinates_df = google_failed_addresses_coordinate_step_df[
        google_failed_addresses_coordinate_step_df['lon'].notna()]
    failed_addresses_coordinate_step_df = google_failed_addresses_coordinate_step_df[
        google_failed_addresses_coordinate_step_df['lon'].isna()]

    nominatim_coordinates_df.to_csv('nominatim_coordinates_df.csv')

    coordinates_df = pandas.concat([google_coordinates_df, nominatim_coordinates_df], ignore_index=True)

    return coordinates_df, failed_addresses_coordinate_step_df


def address_to_coordinates_with_google(row):
    base_url = 'https://www.google.com/search'
    headers = {'User-Agent': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'}
    params = {'source': 'opera', 'q': " ".join([row['Address'], row['City'], row['State'], row['ZIP Code']])}
    base_google_search_request = requests.get(base_url, headers=headers, params=params)
    if base_google_search_request.status_code == 200:
        google_map_urls = re.findall(r'https://maps.google.com/maps\?um=1\S+(?=")', base_google_search_request.text)
        if google_map_urls:
            for google_map_url in google_map_urls:
                coords = re.search(r'(?<=ll=)-?\d{1,2}(\d+|.\d+),-?\d{1,3}(\d+|.\d+)', google_map_url)
                if coords:
                    coords_string = coords.group()
                    return eval(f'({coords_string})')
                else:
                    google_directions_request = requests.get(google_map_url)
                    if google_directions_request.status_code == 200:
                        google_static_image_urls = re.findall(
                            r'https://maps.google.com/maps/api/staticmap\?\S+(?=")', google_directions_request.text)
                        if google_static_image_urls:
                            for google_static_image_url in google_static_image_urls:
                                coords = re.search(r'(?<=markers=)-?\d{1,2}(\d+|.\d+)%2C-?\d{1,2}(\d+|.\d+)(?=&)',
                                                   google_static_image_url)
                                if coords:
                                    coords_string = coords.group()
                                    coords_pair_list = [float(i) for i in coords_string.split('%2C')]
                                    return tuple(coords_pair_list)
    return None, None


def address_to_coordinates_with_nominatim(row, geocode):
    location = geocode(query={'street': row['Address'], 'state': row['State'],
                              'country': 'United States', 'postalcode': row['ZIP Code'][:5]},
                       country_codes='us')
    if location:
        lon, lat = location.longitude, location.latitude
    else:
        lon, lat = None, None
    return lon, lat


### Step 2.4 ###
def second_pass_geocode_using_coordinates(coordinates_df):
    if coordinates_df.empty:
        return coordinates_df, coordinates_df

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


### Step 2.5 ###
def combine_first_and_second_pass_results(first_pass_matched_df, second_pass_matched_df, po_and_route_box_df,
                                          failed_addresses_coordinate_step_df, failed_addresses_geocode_step_df):
    print('Combining successful matches from first and second passes together as well as combining failed addresses '
          'together...')
    addresses_blockgroup_df = pandas.concat([first_pass_matched_df, second_pass_matched_df], ignore_index=True)
    failed_df = pandas.concat([po_and_route_box_df, failed_addresses_coordinate_step_df,
                               failed_addresses_geocode_step_df], ignore_index=True)
    return addresses_blockgroup_df, failed_df


### Step 3 ###

def blockgroup_to_adi(addresses_blockgroup_df):
    print('Adding national and state ADI ranks to the dataframe of successful matches based on Census Block Group '
          'codes...')
    with importlib.resources.path('geocode-adi.resources', 'BlockGroupToADI.feather') as blockgroup_to_adi_feather_path:
        blockgroup_to_adi_df = pandas.read_feather(blockgroup_to_adi_feather_path)
    addresses_blockgroup_df['FIPS'] = addresses_blockgroup_df['FIPS'].str.slice(stop=12)
    addresses_blockgroup_adi_df = addresses_blockgroup_df.merge(blockgroup_to_adi_df, on='FIPS')
    return addresses_blockgroup_adi_df


### Step 4 ###

def export_data(successful_df, failed_df, patient_data_df):
    print('Combining successful and failed addresses back with their original data...')
    successful_df = successful_df.explode('data_id', ignore_index=True).drop(columns='id')
    failed_df = failed_df.explode('data_id', ignore_index=True).drop(columns='id')

    successful_df = patient_data_df.merge(successful_df, on='data_id').drop(columns='data_id')
    failed_df = patient_data_df.merge(failed_df, on='data_id').drop(columns='data_id')

    print('Exporting successful and failed conversion data to CSV...')
    successful_df.to_csv('successful.csv', index=False)
    failed_df.to_csv('failed.csv', index=False)
