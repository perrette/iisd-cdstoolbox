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
# import dimarray as da
import datetime
import cdsapi

from common import (ERA5, CMIP5, Transform, 
    correct_yearly_bias, correct_monthly_bias, convert_time_units_series,
    save_csv, load_csv)
from cmip5 import get_models_per_asset, get_models_per_indicator, get_all_models, cmip5 as cmip5_def


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
    g.add_argument('--period', default='200601-210012', help=argparse.SUPPRESS) # all CMIP5 models and future experiements share the same parameter...
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

        vdef2 = vdef.get('era5',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))
        era5 = ERA5(vdef2.get('name', name), area=o.area, transform=transform, year=o.year, units=vdef['units'], tiled=o.tiled)
        era5.simulation_set = 'ERA5'
        era5.set_folder = 'era5'
        era5.alias = name

        if not o.dataset or o.dataset == 'era5' or o.bias_correction:
            variables.append(era5)

        vdef2 = vdef.get('cmip5',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))

        if not o.dataset or o.dataset == 'cmip5':
            for model in o.model:
                if o.historical:
                    historical = CMIP5(vdef2.get('name', name), model, 'historical', '185001-200512', transform=transform, units=vdef['units'])
                    historical.alias = name
                else:
                    historical = None
                labels = {'rcp_8_5': 'RCP 8.5', 'rcp_4_5': 'RCP 4.5', 'rcp_6_0': 'RCP 6', 'rcp_2_6': 'RCP 2.6'}
                for experiment in o.experiment:
                    cmip5 = CMIP5(vdef2.get('name', name), model, experiment, o.period, transform=transform, units=vdef['units'], historical=historical)
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

            if o.bias_correction and isinstance(v, CMIP5):
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
                    if isinstance(v, ERA5):
                        y1, y2 = o.reference_period
                        roll = False
                        title = f'ERA5: {y1}-{y2}'
                    else:
                        y1, y2 = 2071, 2100
                        roll=True if o.area[1] < 0 else False
                        title = f'{labels.get(v.experiment, v.experiment)} ({v.model}): {y1}-{y2}'

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
                    if isinstance(v, ERA5):
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
