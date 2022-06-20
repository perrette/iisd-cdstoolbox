import os
import cftime
import concurrent.futures
import yaml
import zipfile
import numpy as np
import pandas as pd
import netCDF4 as nc

import cdsapi
from cmip6 import get_all_models


from common import Dataset, CMIP6, time_units


ENSEMBLE_MEMBERS = {
    'cnrm_cm6_1': 'r1i1p1f2',
    'cnrm_cm6_1_hr': 'r1i1p1f2',
    'cnrm_esm2_1': 'r1i1p1f2',
    'miroc_es2l': 'r1i1p1f2',
    'ukesm1_0_ll': 'r1i1p1f2',
}


class SisExtremesIndicesCMIP6(CMIP6):
    pass


class ExtremeValueIndices(SisExtremesIndicesCMIP6):

    def __init__(self, variable, model, experiment, historical=None, frequency=None, date=None, ensemble=None):

        if frequency is None:
            frequency = 'yearly'
        if frequency != 'yearly': raise NotImplementedError("Only yearly frequency is implemented")

        self.frequency = frequency

        ensemble = ensemble or ENSEMBLE_MEMBERS.get(model, "r1i1p1f1")

        assert type(experiment) is str, experiment

        dataset = 'sis-extreme-indices-cmip6'

#         date = ['1900-12-01/2014-12-31'] if experiment == 'historical' else ['2015-01-01/2100-12-31']
        date = ['1850-2014'] if experiment == 'historical' else ['2015-2100']
        if date is str:
            date = [date]

#         datestamp = date[0].split("/")[0].replace("-","") + '-' + date[-1].split("/")[1].replace("-","")
        datestamp = date[0].split("-")[0] + '-' + date[-1].split("-")[-1]

        folder = os.path.join('download', dataset)
        name = f'{variable}-{model}-{experiment}-{ensemble}-{datestamp}'

        downloaded_file = os.path.join(folder, name+'.zip')

        Dataset.__init__(self, dataset,
            {
                'temporal_aggregation': frequency,
                'experiment': experiment,
                'variable': variable,
                'model': model,
                'period': date,
                # 'area': area,
                'ensemble_member': ensemble,
                'format': 'zip',
                'product_type': 'base_independent',
                'ensemble_member': ensemble,
            }, downloaded_file)


        # initialize an `historical` attribute
        if historical is True:
            historical = ExtremeValueIndices(variable, model, "historical", date=date, frequency=frequency, ensemble=ensemble)

        elif historical is False:
            historical = None

        self.historical = historical


VARIABLES =  [
    'consecutive_dry_days', 'consecutive_wet_days', 'diurnal_temperature_range',
    'frost_days', 'growing_season_length', 'heavy_precipitation_days',
    'ice_days', 'maximum_1_day_precipitation', 'maximum_5_day_precipitation',
    'maximum_value_of_daily_maximum_temperature', 'maximum_value_of_daily_minimum_temperature', 'minimum_value_of_daily_maximum_temperature',
    'minimum_value_of_daily_minimum_temperature', 'number_of_wet_days', 'simple_daily_intensity_index',
    'summer_days', 'total_wet_day_precipitation', 'tropical_nights',
    'very_heavy_precipitation_days'
]


# MODELS = get_all_models()
_MODELS_SSP585 = [  'access_cm2', 'access_esm1_5', 'bcc_csm2_mr',
            'ec_earth3', 'ec_earth3_veg', 'gfdl_cm4',
            'gfdl_esm4', 'inm_cm4_8', 'inm_cm5_0',
            'kace_1_0_g', 'kiost_esm', 'miroc6',
            'mpi_esm1_2_hr', 'mpi_esm1_2_lr', 'mri_esm2_0',
            'nesm3', 'noresm2_lm', 'noresm2_mm',] + [  'cnrm_cm6_1', 'cnrm_cm6_1_hr', 'cnrm_esm2_1', 'miroc_es2l', 'ukesm1_0_ll', ]


_MODELS_HISTORICAL = ['access_cm2', 'access_esm1_5', 'bcc_csm2_mr',
            'canesm5', 'ec_earth3', 'ec_earth3_veg',
            'gfdl_cm4', 'gfdl_esm4', 'inm_cm4_8',
            'inm_cm5_0', 'kace_1_0_g', 'kiost_esm',
            'miroc6', 'mpi_esm1_2_hr', 'mpi_esm1_2_lr',
            'mri_esm2_0', 'nesm3', 'noresm2_lm',
            'noresm2_mm',] + [  'cnrm_cm6_1', 'cnrm_cm6_1_hr', 'cnrm_esm2_1', 'miroc_es2l', 'ukesm1_0_ll', ]


_MODELS_EXCLUDE = ['kace_1_0_g'] # netCDF file corrupt

# All models available for the above variables, for both historical and SSP585 experiments, with the r1i1p1f1 ensemble member
MODELS = list(sorted(set(_MODELS_SSP585).intersection(_MODELS_HISTORICAL).difference(_MODELS_EXCLUDE)))

def main():

    import argparse

    locations = yaml.safe_load(open('locations.yml'))
    cmip6_yml = yaml.safe_load(open('cmip6.yml'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel threads for data download. Hint: use `--max-workers 1` for serial downlaod.")
    # g = parser.add_argument_group('variables or asset')
    # g = parser.add_mutually_exclusive_group(required=True)
    # g.add_argument('--era5', nargs='*', help='list of ERA5-monthly variables to download (original name, no correction)')
    # g.add_argument('--cmip6', nargs='*', help='list of CMIP6-monthly variables to download')
    parser.add_argument('--indicator', required=True, choices=VARIABLES)

    # parser.add_argument('--dataset', choices=['era5', 'cmip6'], help='dataset in combination with for `--indicators` and `--asset`')
    parser.add_argument('-o', '--output', default='indicators', help='output directory, default: %(default)s')
    parser.add_argument('--overwrite',action='store_true', help=argparse.SUPPRESS)

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    g = parser.add_argument_group('CMIP6 control')
    g.add_argument('--model', nargs='*', default=None, choices=MODELS)
    g.add_argument('--ensemble_member', default=None, help="typically `r1i1p1f1` but some models require different members")
    g.add_argument('--experiment', nargs='*', choices=cmip6_yml["experiments"], default=['ssp5_8_5'])
    # g.add_argument('--ensemble', action='store_true', help='If `--model` is not specified, default to all available models for the standard set of parameters. ')

    o = parser.parse_args()

    if not o.model:
        o.model = MODELS

    if not (o.location or (o.lon and o.lat)):
        parser.error('please provide a location, for instance `--location Welkenraedt`, or use custom lon and lat, e.g. `--lon 5.94 --lat 50.67`')

    elif o.location:
        loc = {loc['name']: loc for loc in locations}[o.location]
        o.lon, o.lat = loc['lon'], loc['lat']

    print('lon', o.lon)
    print('lat', o.lat)

    for experiment in o.experiment:
        variables = [ExtremeValueIndices(o.indicator, model, experiment, historical=experiment != "historical", ensemble=o.ensemble_member) for model in o.model]

        # https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example
        downloaded_variables = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=o.max_workers) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = { executor.submit(v.download): v for v in variables }
            for future in concurrent.futures.as_completed(future_to_url):
                v = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f'failed to download {v} : {exc}')
                else:
                    downloaded_variables.append(v)


        dataset = {}

        # homogenize units
        dates = np.array([cftime.DatetimeGregorian(y, 12, 31) for y in range(1979,2100+1)])
        index = pd.Index(cftime.date2num(dates, time_units), name=time_units)

        for v in downloaded_variables:
            series = v.load_timeseries(lon=o.lon, lat=o.lat, overwrite=True)
            series.index = index[:len(series)] # otherwise we have things like 180, 182 etc

            dataset[v.model] = series

        df = pd.DataFrame(dataset)

        loc_folder = o.location.lower() if o.location else f'{o.lat}N-{o.lon}E'
        folder = os.path.join(o.output, loc_folder, "extremes")
        csv_file = os.path.join(folder, o.indicator + '.csv')

        os.makedirs(folder, exist_ok=True)

        print("Save to file",csv_file)
        df.to_csv(csv_file)


if __name__ == "__main__":
    main()