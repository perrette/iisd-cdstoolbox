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
import cdsapi

from common import (ERA5, CMIP5, Indicator,
    correct_yearly_bias, correct_monthly_bias, convert_time_units_series, 
    daily_min, daily_max, daily_mean, monthly_mean, yearly_mean, frost_days_per_month,
    save_csv, load_csv, era5_tile_area, make_area, cube_area)

from cmip5 import get_models_per_asset, get_models_per_indicator, get_all_models, cmip5 as cmip5_def

transform_namespace = {
    'daily_min': daily_min, 
    'daily_max': daily_max, 
    'daily_mean': daily_mean, 
    'monthly_mean': monthly_mean, 
    'yearly_mean': yearly_mean,
    'frost_days_per_month': frost_days_per_month,
}

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
        # add indicators ?
        # import indicator
        # kwargs.update(vars(indicator))
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
        print(defs['transform'])
        if type(defs['transform']) is str:
            transforms.append(Transform(transform=defs['transform']))
        else:
            for transform in defs['transform']:
                print(transform)
                transforms.append(Transform(transform=transform))

    transform = SequentialTransform(transforms)

    return cls(vdef['name'], transform=transform, frequency=defs.get('frequency'), **cls_kwargs)


def parse_indicator(cls, name, units=None, description=None, scale=1, offset=0, defs={}, cls_kwargs={}):
    """parse indicator from indicators.yml
    """
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


def main():
    import argparse

    locations = yaml.safe_load(open('locations.yml'))
    variables_def = yaml.safe_load(open('indicators.yml'))
    assets = yaml.safe_load(open('assets.yml'))

    parser = argparse.ArgumentParser()
    # g = parser.add_argument_group('variables or asset')
    g = parser.add_mutually_exclusive_group(required=True)
    # g.add_argument('--era5', nargs='*', help='list of ERA5-monthly variables to download (original name, no correction)')
    # g.add_argument('--cmip5', nargs='*', help='list of CMIP5-monthly variables to download')
    g.add_argument('--indicators', nargs='*', default=[], choices=[vdef['name'] for vdef in variables_def], help='list of custom indicators to download')
    g.add_argument('--asset', choices=list(assets.keys()), help='pre-defined list of variables, defined in assets.yml (experimental)')

    parser.add_argument('--dataset', choices=['era5', 'cmip5'], help='dataset in combination with for `--indicators` and `--asset`')
    parser.add_argument('-o', '--output', default='indicators', help='output directory, default: %(default)s')
    parser.add_argument('--overwrite',action='store_true', help=argparse.SUPPRESS)

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    g = parser.add_argument_group('area size controls')
    g.add_argument('--width-km', type=float, help=argparse.SUPPRESS)
    g.add_argument('--tiled', action='store_true', help=argparse.SUPPRESS)
    g.add_argument('--tile', type=float, nargs=2, default=[10, 5], help=argparse.SUPPRESS)
    #g.add_argument('--tile', type=float, nargs=2, default=[10, 5], help='ERA5 tile in degress lon, lat (%(default)s by default)')
    g.add_argument('--area', nargs=4, type=float, help='area as four numbers: top, left, bottom, right (CDS convention)')

    g = parser.add_argument_group('ERA5 control')
    # g.add_argument('--year', nargs='+', default=list(range(1979, 2019+1)), help='ERA5 years to download, default: %(default)s')
    g.add_argument('--year', nargs='+', default=list(range(1979, 2019+1)), help=argparse.SUPPRESS)

    g = parser.add_argument_group('CMIP5 control')
    g.add_argument('--model', nargs='*', default=['ipsl_cm5a_mr'], choices=get_all_models())
    g.add_argument('--experiment', nargs='*', choices=['rcp_2_6', 'rcp_4_5', 'rcp_6_0', 'rcp_8_5'], default=['rcp_8_5'])
    g.add_argument('--period', default=None, help=argparse.SUPPRESS) # all CMIP5 models and future experiements share the same parameter...
    # g.add_argument('--historical', action='store_true', help='this flag provokes downloading historical data as well and extend back the CMIP5 timeseries to 1979')
    g.add_argument('--historical', action='store_true', default=True, help=argparse.SUPPRESS)
    g.add_argument('--no-historical', action='store_false', dest='historical', help=argparse.SUPPRESS)
    # g.add_argument('--bias-correction', action='store_true', help='align CMIP5 variables with matching ERA5')
    g.add_argument('--bias-correction', action='store_true', default=True, help=argparse.SUPPRESS)
    g.add_argument('--no-bias-correction', action='store_false', dest='bias_correction', help='suppress bias-correction for CMIP5 data')
    g.add_argument('--reference-period', default=[1979, 2019], nargs=2, type=int, help='reference period for bias correction (default: %(default)s)')
    g.add_argument('--yearly-bias', action='store_true', help='yearly instead of monthly bias correction')


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
    if o.area and o.width_km:
        parser.error('only one of --area or --width-km may be provided')

    elif o.location:
        loc = {loc['name']: loc for loc in locations}[o.location]
        o.lon, o.lat = loc['lon'], loc['lat']
        if 'area' in loc and not o.area and not o.width_km:
            o.area = loc['area']

    if o.width_km:
        o.area = make_area(o.lon, o.lat, o.width_km)

    if not o.area:
        dx, dy = o.tile
        if np.mod(360, dx) != 0 or np.mod(180, dy) != 0:
            parser.error('tile size must be a divider of 360, 180')
        o.area = era5_tile_area(o.lon, o.lat, dx, dy)

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

    figures_created = False

    # loop over indicators
    vdef_by_name = {v['name'] : v for v in variables_def}
    for name in o.indicators:

        variables = []  # each variable for the simulation set

        vdef = vdef_by_name[name]
        indicator_def = dict(name=name, units=vdef.get('units'), description=vdef.get('description'), 
            scale=vdef.get('scale', 1), offset=vdef.get('offset', 0))

        vdef2 = vdef.get('era5',{})
        era5_kwargs = dict(area=o.area, year=o.year, tiled=o.tiled)
        era5 = parse_indicator(ERA5, defs=vdef2, cls_kwargs=era5_kwargs, **indicator_def)

        era5.simulation_set = 'ERA5'
        era5.set_folder = 'era5'
        era5.alias = name

        if not o.dataset or o.dataset == 'era5' or o.bias_correction:
            variables.append(era5)

        vdef2 = vdef.get('cmip5',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))

        if not o.dataset or o.dataset == 'cmip5':
            for model in o.model:
                labels = {'rcp_8_5': 'RCP 8.5', 'rcp_4_5': 'RCP 4.5', 'rcp_6_0': 'RCP 6', 'rcp_2_6': 'RCP 2.6'}
                if o.historical:
                    historical_kwargs = dict(model=model, experiment='historical', period=None)
                    historical = parse_indicator(CMIP5, defs=vdef2, cls_kwargs=historical_kwargs, **indicator_def)
                else:
                    historical = None
                for experiment in o.experiment:
                    cmip5_kwargs = dict(model=model, experiment=experiment, period=o.period, historical=historical)
                    cmip5 = parse_indicator(CMIP5, defs=vdef2, cls_kwargs=cmip5_kwargs, **indicator_def)
                    cmip5.reference = era5
                    cmip5.simulation_set = f'CMIP5 - {labels.get(experiment, experiment)} - {model}'
                    cmip5.set_folder = f'cmip5-{model}-{experiment}'
                    cmip5.alias = name
                    variables.append(cmip5)


        if not variables:
            logging.warning(f'no variable for {name}')
            continue

        # download and convert to csv
        for v in variables:
            series = v.load_timeseries(o.lon, o.lat, overwrite=o.overwrite)

            if o.bias_correction and isinstance(v.datasets[0], CMIP5):
                era5 = v.reference.load_timeseries(o.lon, o.lat)
                if o.yearly_bias:
                    series = correct_yearly_bias(series, era5, o.reference_period)
                else:
                    series = correct_monthly_bias(series, era5, o.reference_period)

            folder = os.path.join(o.output, loc_folder, asset_folder, v.set_folder)
            os.makedirs(folder, exist_ok=True)
            v.csv_file = os.path.join(folder, (v.alias or v.variable) + '.csv')
            save_csv(series, v.csv_file)


        if o.view_region or o.view_timeseries or o.png_region or o.png_timeseries:
            import matplotlib.pyplot as plt
            try:
                import cartopy
                import cartopy.crs as ccrs
                kwargs = dict(projection=ccrs.PlateCarree())
            except ImportError:
                logging.warning('install cartopy to benefit from coastlines')
                cartopy = None
                kwargs = {}

            for v in variables:
                v0 = v.datasets[0]
                if figures_created and not (o.view_region or o.view_timeseries):
                    # reuse same figure (speed up)
                    pass
                else:
                    figures_created = True
                    if o.view_region or o.png_region:
                        fig1 = plt.figure()
                        ax1 = plt.subplot(1, 1, 1, **kwargs)

                    if o.view_timeseries or o.png_timeseries:
                        fig2 = plt.figure()
                        ax2 = plt.subplot(1, 1, 1)

                # import view
                if o.view_region or o.png_region:
                    ax1.clear()
                    if not o.view_region and 'cb' in locals(): cb.remove()
                    if isinstance(v.datasets[0], ERA5):
                        y1, y2 = o.reference_period
                        roll = False
                        title = f'ERA5: {y1}-{y2}'
                    else:
                        y1, y2 = 2071, 2100
                        roll=True if o.area[1] < 0 else False
                        title = f'{labels.get(v0.experiment, v0.experiment)} ({v0.model}): {y1}-{y2}'

                    refslice = slice(str(y1), str(y2))
                    map = v.load_cube(time=refslice, area=o.area, roll=roll).mean(dim='time')

                    h = ax1.imshow(map.values[::-1], extent=cube_area(map, extent=True))
                    cb = plt.colorbar(h, ax=ax1, label=f'{name} ({v.units})')
                    # h = map.plot(ax=ax1, cbar_kwargs={'label':f'{v.units}'}, robust=True)
                    ax1.set_title(title)
                    ax1.plot(o.lon, o.lat, 'ko')

                    if cartopy:
                        ax1.coastlines(resolution='10m')

                    if o.png_region:
                        fig1.savefig(v.csv_file.replace('.csv', '-region.png'), dpi=o.dpi)


                if o.view_timeseries or o.png_timeseries:
                    ax2.clear()
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
                        fig2.savefig(v.csv_file.replace('.csv', '.png'), dpi=o.dpi)

            # all simulation sets on one figure
            if o.view_timeseries or o.png_timeseries:
                ax2 = plt.gca()
                ax2.clear()
                for v in variables:
                    ts = load_csv(v.csv_file)
                    ts.index = convert_time_units_series(ts.index, years=True)
                    if isinstance(v.datasets[0], ERA5):
                        color = 'k'
                        zorder = 5
                    else:
                        color = None
                        zorder = None
                    l, = ax2.plot(ts.index, ts.values, alpha=0.5, label=v.simulation_set, linewidth=1, color=color, zorder=zorder)

                    # add yearly mean as well
                    if o.yearly_mean:
                        yearly_mean = ts.rolling(12).mean()
                        l2, = ax2.plot(ts.index[::12], yearly_mean[::12], alpha=1, linewidth=2, color=l.get_color(), zorder=zorder)

                ax2.legend(fontsize='xx-small')
                ax2.set_ylabel(v.units)
                ax2.set_xlabel(ts.index.name)
                ax2.set_title(name)
                # ax2.set_xlim(xmin=start_year, xmax=2100)

                mi, ma = ax2.get_xlim()
                if mi < 0:
                    ax2.set_xlim(xmin=0)  # start at start_year (i.e. ERA5 start)

                if o.png_timeseries:
                    figname = os.path.join(o.output, loc_folder, asset_folder, 'all_'+name+'.png')
                    fig2.savefig(figname, dpi=max(o.dpi, 300))

    if o.view_timeseries or o.view_region:
        plt.show()


if __name__ == '__main__':
    main()
