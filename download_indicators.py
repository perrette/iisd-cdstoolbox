"""Download variables related to one asset class
"""
import os
import zipfile
import logging
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import yaml
# import dimarray as da
import datetime
import cdsapi

from netcdf_to_csv import convert_time

time_units = 'days since 2000-01-01'


class Dataset:
    
    def __init__(self, dataset, params, downloaded_file, transform=None, alias=None, units=None, dataset_alias=None):
        self.dataset = dataset
        self.params = params
        self.downloaded_file = downloaded_file
        self.transform = transform
        self.units = units
        self.alias = alias
        self.dataset_alias = dataset_alias

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        raise AttributeError(name)

    @property
    def folder(self):
        return os.path.dirname(self.downloaded_file)

    @property
    def name(self):
        basename = os.path.basename(self.downloaded_file)
        name, ext = os.path.splitext(basename)
        return name

    def download(self):
        c = cdsapi.Client()

        os.makedirs(self.folder, exist_ok=True)

        print('download...')
        print(self.dataset)
        print(self.params)
        print('>',self.downloaded_file)
        os.makedirs('download', exist_ok=True)

        with open(os.path.join('download', 'log'), 'w+') as f:
            f.write(self.dataset+'\n')
            f.write(str(self.params)+'\n')
            f.write(self.downloaded_file+'\n\n')

        res = c.retrieve(
            self.dataset,
            self.params,
            self.downloaded_file)

        return res


    def __repr__(self):
        return f'{type(self).__name__}({self.dataset}, {self.params}, {self.downloaded_file})'

    def get_varname(self, ds):
        """get main variable name from netCDF.Dataset
        """
        variables = [v for v in ds.variables if ds[v].dimensions == ('time', self.lat, self.lon)]
        assert len(variables) == 1, f'Expected one variable matching (time, {self.lat}, {self.lon}) dimensions, found {len(variables)}.\nVariables: {ds.variables}'
        return variables[0]

    def ncvar(self):
        'main netCDF variable'
        files = self.get_ncfiles()
        with nc.Dataset(files[0]) as ds:
            return self.get_varname(ds)


    def _extract_timeseries(self, f, lon, lat, transform=True):
        if not os.path.exists(self.downloaded_file):
            self.download()
        with nc.Dataset(f) as ds:
            variable = self.get_varname(ds)
            # time = nc.num2date(ds['time'][:], ds['time'].units, ds['time'].calendar)
            time, time_units = convert_time(ds)

        region = xr.open_dataset(f)[variable]
        londim = getattr(region, self.lon)
        latdim = getattr(region, self.lat)
        # is latitude is in reverse order for this dataset ?
        if latdim[1] < latdim[0]:
            interpolator = RegularGridInterpolator((londim, latdim[::-1]), region.values.T[:, ::-1])
        else:
            interpolator = RegularGridInterpolator((londim, latdim), region.values.T)
        timeseries = interpolator(np.array((lon, lat)), method='linear').squeeze()

        units = region.units # file
        series = pd.Series(timeseries, index=time, name=f'{self.variable} ({units})')
        series.index.name = time_units

        if transform:
            series = self._transform_units(series)

        return series


    def extract_timeseries(self, lon, lat, transform=True):
        files = self.get_ncfiles()
        return pd.concat([self._extract_timeseries(f, lon, lat, transform) for f in files])


    def _transform_units(self, series):
        if self.transform:
            series = self.transform(series)
            series.name = f'{self.variable} ({self.units})'

        return series

    def timeseries_file(self, lon, lat):
        base, ext = os.path.splitext(self.downloaded_file)
        return base + f'_{lat}N_{lon}E.csv'

    def load_timeseries(self, lon, lat, overwrite=False):
        '''extract timeseries but buffer file...'''
        fname = self.timeseries_file(lon, lat)
        if not os.path.exists(fname) or overwrite:
            timeseries = self.extract_timeseries(lon, lat, transform=False)
            save_csv(timeseries, fname)

        timeseries = load_csv(fname)
        timeseries = self._transform_units(timeseries)

        return timeseries


    def _extract_map(self, dataarray, time, area=None):
        if time is None:
            time = dataarray.time[-1]        
        map = dataarray.sel(time=time)
        lon = map.coords[self.lon].values
        lat = map.coords[self.lat].values

        # we want to deal with increasing lat            
        if lat[1] < lat[0]: 
            map = map.isel({self.lat:slice(None, None, -1)})
            lat = map.coords[self.lat].values
            assert lat[1] > lat[0]

        if area:
            print('area', area)
            t, l, b, r = area
            ilon = ((lon >= l) & (lon <= r))
            ilat = ((lat <= t) & (lat >= b))
            map = map.isel({self.lon:ilon, self.lat:ilat})
            lon = map.coords[self.lon].values
            lat = map.coords[self.lat].values

        if self.transform:
            map = self.transform(map)

        if self.units:
            map.attrs['units'] = self.units  # enforce user-defined units if defined


        if (lon.size < 2) or (lat.size < 2):
            raise ValueError('region area is too small: point-wise map')

        # add extent as an attribute, for further plotting
        l = lon[0] - (lon[1]-lon[0])/2 # add half a grid cell
        r = lon[-1] + (lon[-1]-lon[-2])/2
        b = lat[0] - (lat[1]-lat[0])/2
        t = lat[-1] + (lat[-1]-lat[-2])/2
        map.attrs['extent'] = np.array((l, r, b, t)).tolist()

        return map


    def extract_map(self, time=None, area=None):
        files = self.get_ncfiles()
        with nc.Dataset(files[0]) as ds:
            variable = self.get_varname(ds)
        if len(files) == 1:
            dataarray = xr.open_dataset(files[0])[variable]
        else:
            dataarray = xr.open_mfdataset(files, combine='by_coords')[variable]
        return self._extract_map(dataarray, time, area)



class CMIP5(Dataset):

    lon = 'lon'
    lat = 'lat'

    def __init__(self, variable, model, experiment, period, ensemble=None, **kwargs):
        if ensemble is None:
            ensemble = 'r1i1p1'

        dataset = 'projections-cmip5-monthly-single-levels'
        folder = os.path.join('download', dataset)
        name = f'{variable}-{model}-{experiment}-{period}-{ensemble}'
        
        downloaded_file = os.path.join(folder, name+'.zip')

        super().__init__(dataset, 
            {
                'variable': variable,
                'model': model,
                'experiment': experiment,
                'period': period,
                'ensemble_member': ensemble,
                'format': 'zip',
            }, downloaded_file, **kwargs)


    def get_ncfiles(self):
        # download zip file
        if not os.path.exists(self.downloaded_file):
            self.download()

        # extract all files if necessary
        with zipfile.ZipFile(self.downloaded_file, 'r') as zipObj:
            listOfiles = zipObj.namelist()

            if not os.path.exists(os.path.join(self.folder, listOfiles[0])):
                print('Extracting all files...')
                zipObj.extractall(path=self.folder)

        return [os.path.join(self.folder, name) for name in listOfiles]


class ERA5(Dataset):

    lon = 'longitude'
    lat = 'latitude'

    def __init__(self, variable, year=None, area=None, **kwargs):
        if area is None:
            area = [90, -180, -90, 180]
        if year is None:
            year = list(range(2000, 2019+1))  # multiple year OK
        dataset = 'reanalysis-era5-single-levels-monthly-means'
        product_type = 'monthly_averaged_reanalysis'
        folder = os.path.join('download', dataset, product_type)
        year0, yearf = year[0], year[-1]
        name = f'{variable}_{year0}-{yearf}_{area[0]}-{area[1]}-{area[2]}-{area[3]}'
        downloaded_file = os.path.join(folder, name+'.nc')

        super().__init__(dataset, {
                'format': 'netcdf',
                'product_type': product_type,
                'variable': variable,
                'year': year,
                'month': list(range(1, 12+1)),
                'time': '00:00',
                'area': area,
            }, downloaded_file, **kwargs)

    def get_ncfiles(self):
        return [self.downloaded_file]



def make_area(lon, lat, w):
    " return `area` keyword top left bottom right for lon/lat and width_km"
    earth_radius = 6371
    latw = np.rad2deg(w/earth_radius)
    disk_radius = earth_radius * np.cos(np.deg2rad(lat))
    lonw = np.rad2deg(w/disk_radius)
    return lat+latw, lon-lonw, lat-latw, lon+lonw


def era5_tile_area(lon, lat, dx=10, dy=5):
    """define a "tile" to re-use some of the files
    """
    # if lon > 180: lon -= 360
    lons = np.arange(0, 360, dx)  # tiles in [0, 360 to match CMIP5]
    lats = np.arange(-90, 90, dy)
    j = np.searchsorted(lons, lon)
    i = np.searchsorted(lats, lat)
    area = lats[i], lons[j-1], lats[i-1], lons[j]
    return np.array(area).tolist() # convert numpy data type to json-compatible python objects


class Transform:
    def __init__(self, scale=1, offset=0, transform=None):
        self.scale = scale
        self.offset = offset
        self.transform = transform

    def __call__(self, data):
        data2 = (data + self.offset)*self.scale
        if self.transform:
            data2 = self.transform(data2)
        return data2


def load_csv(fname):
    return pd.read_csv(fname, index_col=0, comment='#').squeeze()

def save_csv(series, fname):
    series.to_csv(fname)


def main():
    import argparse
    from cmip5 import get_models_per_asset, get_models_per_indicator, get_all_models

    locations = yaml.safe_load(open('locations.yml'))
    variables_def = yaml.safe_load(open('indicators.yml'))
    assets = yaml.safe_load(open('assets.yml'))

    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('variables or asset')
    # g.add_argument('--cmip5', nargs='*', default=[], help='list of CMIP5-monthly variables to download')
    # g.add_argument('--era5', nargs='*', default=[], help='list of ERA5-monthly variables to download')
    g.add_argument('--indicators', nargs='*', default=[], choices=[vdef['name'] for vdef in variables_def], help='list of custom indicators to download')
    g.add_argument('--asset', choices=list(assets.keys()), help='pre-defined list of variables, defined in assets.yml (experimental)')
    g.add_argument('--dataset', choices=['era5', 'cmip5'], help='datasets for --variable and --asset')

    g = parser.add_argument_group('filters (post-processing)')
    g.add_argument('--bias-correction', action='store_true', help='align CMIP5 variables with matching ERA5')
    g.add_argument('--reference-period', default=[2006, 2019], nargs=2, type=int, help='reference period for bias-correction (default: %(default)s)')

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    g = parser.add_argument_group('area size controls')
    g.add_argument('--width-km', type=float, help='width of window (in km) around lon/lat (%(default)s km by default)')
    g.add_argument('-l','--left', type=float)
    g.add_argument('-r','--right', type=float)
    g.add_argument('-b','--bottom', type=float)
    g.add_argument('-t','--top', type=float)

    g = parser.add_argument_group('CMIP5 control')
    g.add_argument('--model', choices=get_all_models())
    default_model = 'ipsl_cm5a_mr'
    g.add_argument('--default-model', action='store_true', help=f'same as --model {default_model}')
    g.add_argument('--experiment', choices=['rcp_2_6', 'rcp_4_5', 'rcp_6_0', 'rcp_8_5'], default='rcp_8_5')
    g.add_argument('--period', default='200601-210012')

    # g = parser.add_argument_group('ERA5 control')
    # g.add_argument('--era5-start', default=2000, type=int, help='default: %(default)s')
    # g.add_argument('--era5-end', default=2019, type=int, help='default: %(default)s')

    g = parser.add_argument_group('visualization')
    g.add_argument('--view-region', action='store_true')
    g.add_argument('--view-timeseries', action='store_true')
    g.add_argument('--view-all', action='store_true')
    g.add_argument('--png-region', action='store_true')
    g.add_argument('--png-timeseries', action='store_true')


    o = parser.parse_args()

    if not (o.location or (o.lon and o.lat)):
        parser.error('please provide a location, for instance `--location Welkenraedt`, or use custom lon and lat, e.g. `--lon 5.94 --lat 50.67`')

    elif o.location:
        loc = {loc['name']: loc for loc in locations}[o.location]
        o.lon, o.lat = loc['lon'], loc['lat']

    if o.width_km or o.left and o.right and o.bottom and o.top:
        if o.left and o.right and o.bottom and o.top:
            area = o.top, o.left, o.bottom, o.right
        else:
            t, l, b, r = make_area(o.lon, o.lat, o.width_km)
            area = o.top or t, o.left or l, o.bottom or b, o.right or r
    else:
        area = era5_tile_area(o.lon, o.lat)


    print('lon', o.lon)
    print('lat', o.lat)

    if not o.asset and not o.indicators:
        parser.error('please provide indicators, for example: `--indicators 2m_temperature` or asset, e.g. `--asset energy`')

    variables = []

    # assets only contain indicators
    if o.asset:
        for vname in assets[o.asset]:
            if vname not in [v['name'] for v in variables_def]:
                parser.error(f'unknown indicator in assets.yml: {vname}. See indicators.yml for indicator definition')
            o.indicators.append(vname)

    # add indicators
    vdef_by_name = {v['name'] : v for v in variables_def}
    for name in o.indicators:
        vdef = vdef_by_name[name]

        vdef2 = vdef.get('era5',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))
        era5 = ERA5(vdef2.get('name', name), area=area, transform=transform, units=vdef['units'], alias=name)

        if not o.dataset or o.dataset == 'era5' or o.bias_correction:
            variables.append(era5)

        vdef2 = vdef.get('cmip5',{})
        transform = Transform(vdef2.get('scale', 1), vdef2.get('offset', 0))

        if not o.dataset or o.dataset == 'cmip5':
            if o.model:
                models = [o.model]
            elif o.default_model:
                models = [default_model]
            else:
                if o.asset:
                    models = get_models_per_asset(o.asset, experiment=o.experiment)
                else:
                    models = get_models_per_indicator(name, experiment=o.experiment)

            for model in models:
                cmip5 = CMIP5(vdef2.get('name', name), o.model, o.experiment, o.period, transform=transform, units=vdef['units'], alias=name)
                cmip5.reference = era5
                variables.append(cmip5)

    # folder structure for CSV results
    loc_folder = o.location.lower() if o.location else f'{o.lat}N-{o.lon}E' 
    asset_folder = o.asset if o.asset else 'all'

    # download and convert to csv
    for v in variables:
        # if not os.path.exists(v.downloaded_file):
            # v.download()
        series = v.load_timeseries(o.lon, o.lat)

        # additional transform for bias correction
        if o.bias_correction and isinstance(v, CMIP5):
            # ERA5 was already loaded
            era5 = v.reference.load_timeseries(o.lon, o.lat)
            y1, y2 = o.reference_period
            t1 = nc.date2num(datetime.datetime(y1, 1,1), time_units)
            t2 = nc.date2num(datetime.datetime(y2, 12,12), time_units)
            climatology = era5.loc[t1:t2].mean()
            ref = series.loc[t1:t2].mean()
            delta = climatology - ref
            print(f'apply bias correction to {v.dataset}, {v.variable}: {delta}')
            series = series + delta


        if isinstance(v, ERA5):
            set_folder = 'era5'
        elif isinstance(v, CMIP5):
            set_folder = f'cmip5-{o.model}-{o.experiment}'
        else:
            raise ValueError(repr(v))

        folder = os.path.join('indicators', loc_folder, asset_folder, set_folder)
        os.makedirs(folder, exist_ok=True)
        v.csv_file = os.path.join(folder, (v.alias or v.variable) + '.csv')
        save_csv(series, v.csv_file)


    if (o.png_region or o.png_timeseries) and o.view_all:
        logging.warning('--view-all is not possible together with --png flags')
        o.view_all = False

    if o.view_region or o.view_timeseries or o.png_region or o.png_timeseries or o.view_all:
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
            if o.view_all:
                fig = plt.figure()
                ax1 = plt.subplot(2, 1, 1, **kwargs)
                ax2 = plt.subplot(2, 1, 2)
                o.view_region = True
                o.view_timeseries = True

            else:
                if o.view_region or o.png_region:
                    fig1 = plt.figure()
                    ax1 = plt.subplot(1, 1, 1, **kwargs)

                if o.view_timeseries or o.png_timeseries:
                    fig2 = plt.figure()
                    ax2 = plt.subplot(1, 1, 1)

            # import view
            if o.view_region or o.png_region:
                try:
                    map = v.extract_map(area=area)
                    h = ax1.imshow(map.values[::-1], extent=map.extent)
                    plt.colorbar(h, ax=ax1, label=f'{v.variable} ({map.units})')
                    ax1.set_title(v.dataset)
                    ax1.plot(o.lon, o.lat, 'ko')

                    if cartopy:
                        ax1.coastlines(resolution='10m')

                    if o.png_region:
                        fig1.savefig(v.csv_file.replace('.csv', '-region.png'))
                    if not o.view_region:
                        plt.close(fig1)
                except:
                    raise
                    pass


            if o.view_timeseries or o.png_timeseries:
                ts = v.load_timeseries(o.lon, o.lat)
                # convert units for easier reading of graphs
                ts.index = ts.index / 365.25 + 2000
                ts.index.name = 'years since 2000-01-01'
                ts.plot(ax=ax2)
                ax2.set_ylabel(v.units)
                ax2.set_title(v.dataset)

                if o.png_timeseries:
                    fig2.savefig(v.csv_file.replace('.csv', '.png'))
                if not o.view_timeseries:
                    plt.close(fig2)

        plt.show()


if __name__ == '__main__':
    main()
