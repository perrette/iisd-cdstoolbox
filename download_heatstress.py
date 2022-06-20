import os
import cftime
import concurrent.futures
import yaml
import zipfile
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xa

import cdsapi
from cmip6 import get_all_models


from common import Dataset, CMIP6, time_units
from download_extremes import ENSEMBLE_MEMBERS, SisExtremesIndicesCMIP6


class HeatStress(SisExtremesIndicesCMIP6):

    def __init__(self, variable, model, experiment, ensemble=None):

        ensemble = ensemble or ENSEMBLE_MEMBERS.get(model, "r1i1p1f1")

        assert type(experiment) is str, experiment

        dataset = 'sis-extreme-indices-cmip6'

#         date = ['1900-12-01/2014-12-31'] if experiment == 'historical' else ['2015-01-01/2100-12-31']
        date = ['20110101-21001231']

#         datestamp = date[0].split("/")[0].replace("-","") + '-' + date[-1].split("/")[1].replace("-","")
        datestamp = date[0].split("-")[0] + '-' + date[-1].split("-")[-1]

        folder = os.path.join('download', dataset)
        name = f'{variable}-{model}-{experiment}-{ensemble}-{datestamp}'

        downloaded_file = os.path.join(folder, name+'.zip')

        Dataset.__init__(self, dataset,
            {
                'temporal_aggregation': 'daily',
                'experiment': experiment,
                'variable': variable,
                'model': model,
                'period': date,
                # 'area': area,
                'ensemble_member': ensemble,
                'format': 'zip',
                'product_type': 'bias_adjusted',
            }, downloaded_file)

        self.historical = None



VARIABLES =  [
            'heat_index', 'humidex', 'universal_thermal_climate_index',
            'wet_bulb_globe_temperature_index', 'wet_bulb_temperature_index',
]


# MODELS = get_all_models()
_MODELS_SSP585 = [  'access_cm2', 'access_esm1_5', 'canesm5',
            'ec_earth3', 'ec_earth3_veg', 'fgoals_g3',
            'gfdl_cm4', 'gfdl_esm4', 'inm_cm4_8',
            'inm_cm5_0', 'kiost_esm', 'mpi_esm1_2_hr',
            'mpi_esm1_2_lr', 'mri_esm2_0', 'noresm2_lm',
            'noresm2_mm', ] + [  'cnrm_cm6_1', 'cnrm_cm6_1_hr', 'cnrm_esm2_1', 'miroc_es2l']


_MODELS_EXCLUDE = []

# All models available for the above variables, for both historical and SSP585 experiments, with the r1i1p1f1 ensemble member
MODELS = list(sorted(set(_MODELS_SSP585).difference(_MODELS_EXCLUDE)))

def main():

    import argparse

    locations = yaml.safe_load(open('locations.yml'))

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
    parser.add_argument('--frequency', choices=["daily", "monthly"], default="daily")

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    g = parser.add_argument_group('CMIP6 control')
    g.add_argument('--model', nargs='+', default=None, choices=MODELS)
    g.add_argument('--ensemble_member', default=None, help="typically `r1i1p1f1` but some models require different members")
    g.add_argument('--experiment', nargs='*', choices=['ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp5_8_5'], default=['ssp5_8_5'])
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
        variables = [HeatStress(o.indicator, model, experiment, ensemble=o.ensemble_member) for model in o.model]

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
        if o.frequency == "monthly":
            end_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            dates = np.array([cftime.DatetimeGregorian(y, m, end_of_month[m-1]) for y in range(1979,2100+1) for m in range(1, 12+1)])
            index = pd.Index(cftime.date2num(dates, time_units), name=time_units)

        loc_folder = o.location.lower() if o.location else f'{o.lat}N-{o.lon}E'
        folder = os.path.join(o.output, loc_folder, "extremes")
        os.makedirs(folder, exist_ok=True)

        for v in downloaded_variables:
            series = v.load_timeseries(lon=o.lon, lat=o.lat, overwrite=o.overwrite)

            # monthly mean
            if o.frequency == "monthly":
                xa_series = xa.DataArray(series, coords={"time": cftime.num2date(series.index, time_units)}, dims=["time"])
                series = xa_series.resample({"time": "M"}).mean().to_pandas()
                # homogeneize index across calendars
                series.index = index[:len(series)]
                # series.index = pd.Index(cftime.date2num(series.index, time_units), name=time_units)

            csv_file = os.path.join(folder, f'{o.indicator}-{o.frequency}-{v.model}.csv')
            print("Save to file",csv_file)
            series.to_csv(csv_file)

            dataset[v.model] = series

        df = pd.DataFrame(dataset)

        csv_file = os.path.join(folder, f'{o.indicator}-{o.frequency}-all.csv')
        print("Save to file",csv_file)
        df.to_csv(csv_file)


if __name__ == "__main__":
    main()