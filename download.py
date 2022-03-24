"""Download variables related to one asset class
"""
import os
import zipfile
import logging
import itertools
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import yaml
import datetime
import concurrent.futures
import cdsapi

from common import (ERA5, CMIP6, Indicator,
    correct_yearly_bias, correct_monthly_bias, convert_time_units_series,
    save_csv, load_csv, make_area, cube_area, time_units)

import transform

from cmip6 import get_all_models

transform_namespace = { name: getattr(transform, name) for name in transform.__all__ }


class Transform:
    def __init__(self, scale=1, offset=0, transform=None):
        self.scale = scale
        self.offset = offset
        if type(transform) is str:
            if transform not in transform_namespace:
                raise ValueError(f'{transform}. Valid transforms are: {", ".join(transform_namespace)}')
            transform = transform_namespace[transform]  # evaluate daily_min, etc..
        self.transform = transform

    def __call__(self, data):
        result = (data + self.offset)*self.scale
        if self.transform:
            result = self.transform(result)
        return result


class SequentialTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for f in self.transforms:
            data = f(data)
        return data

class ComposeIndicator(Indicator):
    """an indicator defined via "compose" and "expression fields"
    """
    def __init__(self, name, units, description, datasets, expression, mapping=None, **kwargs):
        self.expression = expression
        self.mapping = mapping or {}
        super().__init__(name, units, description, datasets, self._compose, **kwargs)

    def _compose(self, *values):
        """evaluate expression (prefix leading digits with _)
        """
        kwargs = {'_'+dataset.variable if dataset.variable.startswith(tuple(str(i) for i in range(10))) else dataset.variable: value for dataset, value in zip(self.datasets, values)}
        # add numpy function (exp etc...)
        kwargs.update(vars(np))
        # kwargs["exp"] = np.exp
        # kwargs["log"] = np.log
        # kwargs["sqrt"] = np.sqrt

        # allow aliases
        for short, expression in self.mapping.items():
            assert short not in kwargs, f'{short} already exists, cannot be used as alias (mapping)'
            kwargs[short] = eval(expression, kwargs)
        return eval(self.expression, kwargs)


def parse_dataset(cls, name, scale=1, offset=0, defs={}, cls_kwargs={}):
    vdef = {'name': name, 'scale': scale, 'offset': offset}
    vdef.update(defs) # update with dataset-specific values

    transforms = [ Transform(vdef.get('scale', 1), vdef.get('offset', 0)) ]
    if 'transform' in defs:
        if type(defs['transform']) is str:
            transforms.append(Transform(transform=defs['transform']))
        else:
            for transform in defs['transform']:
                transforms.append(Transform(transform=transform))

    transform = SequentialTransform(transforms)

    return cls(vdef['name'], transform=transform, frequency=defs.get('frequency'), **cls_kwargs)


def parse_indicator(cls, name, units=None, description=None, scale=1, offset=0, defs={}, cls_kwargs={}):
    """parse indicator from indicators.yml
    """
    defs = defs.copy() # otherwise any update to defs propagates to the caller
    if 'compose' in defs:
        assert 'expression' in defs, f'{name}: expression must be provided for composed indicator (indicators.yml)'
        assert 'name' not in defs, f'{name}: cannot provide both "compose" and "name" (indicators.yml)'
        assert type(defs['compose']) is list, f'{name}: expected list of variable for "compose" field, got: {type(defs["compose"])}'
        datasets = []
        for name2 in defs['compose']:
            defs.update({'name': name2})
            dataset = parse_dataset(cls, name, scale, offset, defs, cls_kwargs)
            datasets.append(dataset)
        return ComposeIndicator(name, units, description, datasets=datasets, expression=defs['expression'], mapping=defs.get('mapping'))

    dataset = parse_dataset(cls, name, scale, offset, defs, cls_kwargs)
    return Indicator(name, units, description, datasets=[dataset])


def download_all_variables(variables, max_workers=4):
    # https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example
    downloaded_variables = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    return downloaded_variables

def download_all_variables_serial(variables):
    "original version of download_all_variables, without concurrent.futures"
    downloaded_variables = []
    for v in variables:
        try:
            v.download()
        except Exception as error:
            print(error)
            logging.warning(f'failed to download {v}')
            continue
        downloaded_variables.append(v)
    return downloaded_variables


def main():
    import argparse

    locations = yaml.safe_load(open('locations.yml'))
    variables_def = yaml.safe_load(open('indicators.yml'))
    assets = yaml.safe_load(open('assets.yml'))
    cmip6_yml = yaml.safe_load(open('cmip6.yml'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel threads for data download. Hint: use `--max-workers 1` for serial downlaod.")
    # g = parser.add_argument_group('variables or asset')
    g = parser.add_mutually_exclusive_group(required=True)
    # g.add_argument('--era5', nargs='*', help='list of ERA5-monthly variables to download (original name, no correction)')
    # g.add_argument('--cmip6', nargs='*', help='list of CMIP6-monthly variables to download')
    g.add_argument('--indicators', nargs='*', default=[], choices=[vdef['name'] for vdef in variables_def], help='list of custom indicators to download')
    g.add_argument('--asset', choices=list(assets.keys()), help='pre-defined list of variables, defined in assets.yml (experimental)')

    parser.add_argument('--dataset', choices=['era5', 'cmip6'], help='dataset in combination with for `--indicators` and `--asset`')
    parser.add_argument('-o', '--output', default='indicators', help='output directory, default: %(default)s')
    parser.add_argument('--overwrite',action='store_true', help=argparse.SUPPRESS)

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    g = parser.add_argument_group('area size controls')
    g.add_argument('--area', nargs=4, type=float, help='area as four numbers: top, left, bottom, right (CDS convention)')
    g.add_argument('--width-km', type=float, default=1000, help="Width (km) around the selected location, when not provided by `area`. %(default)s km by default.")
    g.add_argument('--view', nargs=4, type=float, help='area for plot as four numbers: top, left, bottom, right (CDS convention)')

    g = parser.add_argument_group('ERA5 control')
    # g.add_argument('--year', nargs='+', default=list(range(1979, 2019+1)), help='ERA5 years to download, default: %(default)s')
    g.add_argument('--year', nargs='+', default=list(range(1979, 2019+1)), help=argparse.SUPPRESS)

    g = parser.add_argument_group('CMIP6 control')
    g.add_argument('--model', nargs='*', default=None, choices=get_all_models())
    g.add_argument('--experiment', nargs='*', choices=cmip6_yml["experiments"], default=['ssp5_8_5'])
    # g.add_argument('--period', default=None, help=argparse.SUPPRESS) # all CMIP6 models and future experiements share the same parameter...
    # g.add_argument('--historical', action='store_true', help='this flag provokes downloading historical data as well and extend back the CMIP6 timeseries to 1979')
    g.add_argument('--historical', action='store_true', default=True, help=argparse.SUPPRESS)
    g.add_argument('--no-historical', action='store_false', dest='historical', help=argparse.SUPPRESS)
    # g.add_argument('--bias-correction', action='store_true', help='align CMIP6 variables with matching ERA5')
    g.add_argument('--bias-correction', action='store_true', default=True, help=argparse.SUPPRESS)
    g.add_argument('--no-bias-correction', action='store_false', dest='bias_correction', help='suppress bias-correction for CMIP6 data')
    g.add_argument('--reference-period', default=[1979, 2019], nargs=2, type=int, help='reference period for bias correction (default: %(default)s)')
    g.add_argument('--yearly-bias', action='store_true', help='yearly instead of monthly bias correction')
    g.add_argument('--ensemble', action='store_true', help='If `--model` is not specified, default to all available models. Also write a csv file with all models as columns, as well as median, lower and upper (5th and 95th percentiles) fields.')


    g = parser.add_argument_group('visualization')
    g.add_argument('--view-region', action='store_true')
    g.add_argument('--view-timeseries', action='store_true')
    g.add_argument('--png-region', action='store_true')
    g.add_argument('--png-timeseries', action='store_true')
    g.add_argument('--dpi', default=100, type=int, help='dop-per-inches (default: %(default)s)')
    g.add_argument('--yearly-mean', action='store_true')


    o = parser.parse_args()

    if not (o.location or (o.lon and o.lat)):
        parser.error('please provide a location, for instance `--location Welkenraedt`, or use custom lon and lat, e.g. `--lon 5.94 --lat 50.67`')

    elif o.location:
        loc = {loc['name']: loc for loc in locations}[o.location]
        o.lon, o.lat = loc['lon'], loc['lat']
        if 'area' in loc and not o.area:
            o.area = loc['area']


    if not o.area:
        o.area = make_area(o.lon, o.lat, o.width_km)

    print('lon', o.lon)
    print('lat', o.lat)

    if not o.asset and not o.indicators:
        parser.error('please provide indicators, for example: `--indicators 2m_temperature` or asset, e.g. `--asset energy`')


    # assets only contain indicators
    if o.asset:
        for vname in assets[o.asset]:
            if vname not in [v['name'] for v in variables_def]:
                parser.error(f'unknown indicator in assets.yml: {vname}. See indicators.yml for indicator definition')
            o.indicators.append(vname)

    # folder structure for CSV results
    loc_folder = o.location.lower() if o.location else f'{o.lat}N-{o.lon}E' 
    asset_folder = o.asset if o.asset else 'all'

    if o.model is None:
        if o.ensemble:
            o.model = get_all_models()
        else:
            o.model = 'mpi_esm1_2_lr'

    # loop over indicators
    vdef_by_name = {v['name'] : v for v in variables_def}
    for name in o.indicators:

        variables = []  # each variable for the simulation set

        vdef = vdef_by_name[name]
        indicator_def = dict(name=name, units=vdef.get('units'), description=vdef.get('description'), 
            scale=vdef.get('scale', 1), offset=vdef.get('offset', 0))

        vdef2 = vdef.get('era5',{})
        era5_kwargs = dict(area=o.area, year=o.year)
        era5 = parse_indicator(ERA5, defs=vdef2, cls_kwargs=era5_kwargs, **indicator_def)

        era5.simulation_set = 'ERA5'
        era5.set_folder = 'era5'
        era5.alias = name

        if not o.dataset or o.dataset == 'era5' or o.bias_correction:
            variables.append(era5)

        vdef2 = vdef.get('cmip6',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))

        if not o.dataset or o.dataset == 'cmip6':
            for model in o.model:
                labels = {x: "{}-{}.{}".format(*x.split("_")) for x in cmip6_yml["experiments"]}
                # if o.historical:
                #     historical_kwargs = dict(model=model, experiment='historical')
                #     historical = parse_indicator(CMIP6, defs=vdef2, cls_kwargs=historical_kwargs, **indicator_def)
                # else:
                #     historical = None
                for experiment in o.experiment:
                    cmip6_kwargs = dict(model=model, experiment=experiment, historical=o.historical, area=o.area)
                    cmip6 = parse_indicator(CMIP6, defs=vdef2, cls_kwargs=cmip6_kwargs, **indicator_def)
                    cmip6.reference = era5
                    cmip6.simulation_set = f'CMIP6 - {labels.get(experiment, experiment)} - {model}'
                    cmip6.set_folder = f'cmip6-{model}-{experiment}'
                    cmip6.alias = name
                    # print("indicator variable", experiment, [d.name for d in cmip6.datasets])
                    variables.append(cmip6)


        if not variables:
            logging.warning(f'no variable for {name}')
            continue

        if o.max_workers < 2:
            variables2 = download_all_variables_serial(variables)
        else:
            variables2 = download_all_variables(variables)

        # Diagnose which variables have been excluded
        names = list(set([v.name for v in variables]))
        names2 = list(set([v.name for v in variables2]))

        models = list(set([v.datasets[0].model for v in variables if isinstance(v.datasets[0], CMIP6)]))
        models2 = list(set([v.datasets[0].model for v in variables2 if isinstance(v.datasets[0], CMIP6)]))

        print(f"Downloaded {len(variables2)} out of {len(variables)}")
        print(f"... {len(names2)} out of {len(names)} variable types")
        print(f"... {len(models2)} out of {len(models)} models")
        print("CMIP6 models excluded:", " ".join([m for m in models if m not in models2]))
        print("CMIP6 models included:", " ".join(models2))

        variables = variables2

        # download and convert to csv
        for v in variables:
            folder = os.path.join(o.output, loc_folder, asset_folder, v.set_folder)
            v.csv_file = os.path.join(folder, (v.alias or v.variable) + '.csv')

            if os.path.exists(v.csv_file):
                print("Already exitst:",v.csv_file)
                continue

            series = v.load_timeseries(o.lon, o.lat, overwrite=o.overwrite)

            bias_correction_method = vdef.get('bias-correction')

            if o.bias_correction and isinstance(v.datasets[0], CMIP6) and bias_correction_method is not None:
                era5 = v.reference.load_timeseries(o.lon, o.lat)
                #v.set_folder += '-unbiased'
                if o.yearly_bias:
                    series = correct_yearly_bias(series, era5, o.reference_period, bias_correction_method)
                else:
                    series = correct_monthly_bias(series, era5, o.reference_period, bias_correction_method)

            os.makedirs(folder, exist_ok=True)
            print("Save to",v.csv_file)
            save_csv(series, v.csv_file)


        if o.ensemble:
            ensemble_files = {}
            import cftime, datetime
            for experiment in o.experiment:
                ensemble_variables = [v for v in variables if isinstance(v.datasets[0], CMIP6) and v.datasets[0].experiment == experiment]
                dates = np.array([cftime.DatetimeGregorian(y, m, 15) for y in range(1979,2100+1) for m in range(1,12+1)])
                index = pd.Index(cftime.date2num(dates, time_units), name=time_units)

                df = {}
                for v in ensemble_variables:
                    series = load_csv(v.csv_file)
                    series.index = index[:len(series)]
                    df[v.datasets[0].model] = series
                df = pd.DataFrame(df)
                median = df.median(axis=1)
                lower = df.quantile(.05, axis=1)
                upper = df.quantile(.95, axis=1)
                df["median"] = median
                df["lower"] = lower
                df["upper"] = upper
                first = ensemble_variables[0]
                folder = os.path.join(o.output, loc_folder, asset_folder, first.set_folder.replace(first.datasets[0].model, "ensemble"))
                csv_file = os.path.join(folder, first.alias or first.name)  + '.csv'
                ensemble_files[experiment] = csv_file
                os.makedirs(folder, exist_ok=True)
                print("Save to",csv_file)
                save_csv(df, csv_file)

        if o.view_region or o.view_timeseries or o.png_region or o.png_timeseries:
            import matplotlib.pyplot as plt
            cb = None
            try:
                import cartopy
                import cartopy.crs as ccrs
                kwargs = dict(projection=ccrs.PlateCarree())
            except ImportError:
                logging.warning('install cartopy to benefit from coastlines')
                cartopy = None
                kwargs = {}

            if o.view is None:
                o.view = o.area

            def plot_timeseries(v):
                figname = v.csv_file.replace('.csv', '.png')
                if os.path.exists(figname):
                    return

                fig2 = plt.figure(num=2)
                plt.clf()
                ax2 = fig2.add_subplot(1, 1, 1)

                ts = load_csv(v.csv_file)
                # convert units for easier reading of graphs
                ts.index = convert_time_units_series(ts.index, years=True)
                # ts.plot(ax=ax2, label=v.simulation_set)
                l, = ax2.plot(ts.index, ts.values, label=v.simulation_set)
                ax2.legend()
                ax2.set_xlabel(ts.index.name)
                ax2.set_ylabel(v.units)
                ax2.set_title(name)

                # add yearly mean as well
                if o.yearly_mean:
                    yearly_mean = ts.rolling(12).mean()
                    l2, = ax2.plot(ts.index[::12], yearly_mean[::12], alpha=1, linewidth=2, color=l.get_color())

                if o.png_timeseries:
                    fig2.savefig(figname, dpi=o.dpi)

            def plot_region(v):
                v0 = v.datasets[0]

                figname = v.csv_file.replace('.csv', '-region.png')
                if os.path.exists(figname):
                    return

                fig1 = plt.figure(num=1)
                plt.clf()
                ax1 = fig1.add_subplot(1, 1, 1, **kwargs)

                if isinstance(v.datasets[0], ERA5):
                    y1, y2 = o.reference_period
                    roll = False
                    title = f'ERA5: {y1}-{y2}'
                else:
                    y1, y2 = 2071, 2100
                    roll=True if o.view[1] < 0 else False
                    title = f'{labels.get(v0.experiment, v0.experiment)} ({v0.model}): {y1}-{y2}'

                refslice = slice(str(y1), str(y2))
                map = v.load_cube(time=refslice, area=o.view, roll=roll).mean(dim='time')

                h = ax1.imshow(map.values[::-1], extent=cube_area(map, extent=True))
                cb = plt.colorbar(h, ax=ax1, label=f'{name} ({v.units})')
                # h = map.plot(ax=ax1, cbar_kwargs={'label':f'{v.units}'}, robust=True)
                ax1.set_title(title)
                ax1.plot(o.lon, o.lat, 'ko')

                if cartopy:
                    ax1.coastlines(resolution='10m')

                if o.png_region:
                    fig1.savefig(figname, dpi=o.dpi)


            for v in variables:


                if o.view_timeseries or o.png_timeseries:
                    plot_timeseries(v)
                

                if o.view_region or o.png_region:
                    try:
                        plot_region(v)
                    except:
                        logging.warning(f'failed to make map for {v.name}')



            # all simulation sets on one figure
            def plot_all_simulations():
                figname = os.path.join(o.output, loc_folder, asset_folder, 'all_'+name+'.png')
                if os.path.exists(figname):
                    return

                fig3 = plt.figure(num=3)
                plt.clf()
                ax3 = fig3.add_subplot(1, 1, 1)
                for v in variables:
                    ts = load_csv(v.csv_file)
                    ts.index = convert_time_units_series(ts.index, years=True)
                    if isinstance(v.datasets[0], ERA5):
                        color = 'k'
                        zorder = 5
                    else:
                        color = None
                        zorder = None

                    # add yearly mean instead of monthly mean
                    if o.yearly_mean:
                        yearly_mean = ts.rolling(12).mean()
                        x = ts.index[::12]
                        y = yearly_mean[::12]
                    else:
                        x = ts.index
                        y = ts.values

                    l, = ax3.plot(x, y, alpha=0.5 if o.ensemble else 1, label=v.simulation_set, linewidth=1 if o.ensemble else 2, color=color, zorder=zorder)

                # Add ensemble mean
                if o.ensemble:
                    for experiment in ensemble_files:
                        df = load_csv(ensemble_files[experiment])
                        df.index = convert_time_units_series(df.index, years=True)

                        if o.yearly_mean:
                            yearly_mean = df.rolling(12).mean()
                            x = df.index[::12]
                            y = yearly_mean.iloc[::12]
                        else:
                            x = df.index
                            y = df

                        l, = ax3.plot(x, y["median"], alpha=1, label=f"{experiment} (median)", linewidth=2, zorder=4)
                        ax3.plot(x, y["lower"], linewidth=1, zorder=4, linestyle="--", color=l.get_color())
                        ax3.plot(x, y["upper"], linewidth=1, zorder=4, linestyle="--", color=l.get_color())
                        ax3.fill_between(x, y["lower"], y["upper"], alpha=0.2, zorder=-1, color=l.get_color())

                ax3.legend(fontsize='xx-small')
                ax3.set_ylabel(v.units)
                ax3.set_xlabel(ts.index.name)
                ax3.set_title(name)
                # ax3.set_xlim(xmin=start_year, xmax=2100)

                mi, ma = ax3.get_xlim()
                if mi < 0:
                    ax3.set_xlim(xmin=0)  # start at start_year (i.e. ERA5 start)

                if o.png_timeseries:
                    fig3.savefig(figname, dpi=max(o.dpi, 300))

            if o.view_timeseries or o.png_timeseries:
                plot_all_simulations()

    if o.view_timeseries or o.view_region:
        plt.show()


if __name__ == '__main__':
    main()
