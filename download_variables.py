"""Download variables related to one asset class
"""
import os
import zipfile
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import yaml
# import dimarray as da
import cdsapi

from netcdf_to_csv import convert_time


class Dataset:
    
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


    def csvfile(self, lon, lat):
        file = os.path.basename(self.downloaded_file)
        path = os.path.dirname(self.downloaded_file)
        path = path.replace('download', 'csv')
        file, ext = os.path.splitext(file)
        return os.path.join(path, file+f'_{lon}E_{lat}N'+'.csv')


    def save_csv(self, lon, lat, fname=None):
        if fname is None:
            fname = self.csvfile(lon, lat)
        series = self.extract_timeseries(lon, lat)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        series.to_csv(fname)


    def __repr__(self):
        return f'{type(self).__name__}({self.dataset}, {self.params}, {self.downloaded_file})'


class CMIP5(Dataset):

    def __init__(self, variable, model, scenario, period, ensemble=None):
        self.variable = variable
        self.model = model
        self.scenario = scenario
        self.period = period
        self.ensemble = ensemble or 'r1i1p1'

        self.dataset = 'projections-cmip5-monthly-single-levels'

        self.folder = os.path.join('download', self.dataset)
        self.name = '{variable}-{model}-{scenario}-{period}-{ensemble}'.format(**vars(self)) 
        self.downloaded_file = os.path.join(self.folder, self.name+'.zip')

    @property
    def params(self):
        return {
                'ensemble_member': self.ensemble,
                'format': 'zip',
                'experiment': self.scenario,
                'variable': self.variable,
                'model': self.model,
                'period': self.period,
            }

    def get_varname(self, ds):
        """get main variable name from netCDF.Dataset
        """
        variables = [v for v in ds.variables if ds[v].dimensions == ('time', 'lat', 'lon')]
        assert len(variables) == 1, f'Expected one variable matching (time, lat, lon) dimensions, found {len(variables)}.\nVariables: {ds.variables}'
        return variables[0]


    def extract_timeseries(self, lon, lat):
        # download zip file
        if not os.path.exists(self.downloaded_file):
            self.download()

        # extract all files if necessary
        with zipfile.ZipFile(self.downloaded_file, 'r') as zipObj:
            listOfiles = zipObj.namelist()

            if not os.path.exists(os.path.join(self.folder, listOfiles[0])):
                print('Extracting all files...')
                zipObj.extractall(path=self.folder)

        # return dataset 
        files = [os.path.join(self.folder, name) for name in listOfiles]

        # alltimes, allvalues = zip(*[self._extract_timeseries(f, lon, lat) for f in files])
        return pd.concat([self._extract_timeseries(f, lon, lat) for f in files])


    def _extract_timeseries(self, f, lon, lat):
        # variable name
        with nc.Dataset(f) as ds:
            variable = self.get_varname(ds)
            time, units = convert_time(ds)

        # load multiple files (rearrange along coordinates)
        da = xr.open_dataset(f)[variable]
        # extract a region around the point of interest
        region = da.sel(lon=slice(lon-5, lon+5), lat=slice(lat-5, lat+5))
        # bi-linear interpolation on the specific lon/lat
        interpolator = RegularGridInterpolator((region.lon, region.lat), region.values.T)
        values = interpolator(np.array((lon, lat)), method='linear').squeeze()
        s = pd.Series(values, index=time, name=self.variable)
        s.index.name = units
        return s


class ERA5(Dataset):

    def __init__(self, variable, year=None, area=None):
        self.variable = variable
        self.area = area or [90, -180, -90, 180]
        self.year = year or list(range(2000, 2019+1))  # multiple year OK

        self.month = list(range(1, 12+1))
        self.dataset = 'reanalysis-era5-single-levels-monthly-means'
        self.product_type = 'monthly_averaged_reanalysis'

        self.folder = os.path.join('download', self.dataset, self.product_type)
        self.name = '{variable}_{year0}-{yearf}_{area[0]}-{area[1]}-{area[2]}-{area[3]}'.format(year0=self.year[0], yearf=self.year[-1], **vars(self))
        self.downloaded_file = os.path.join(self.folder, self.name+'.nc')

    @property
    def params(self):
        return {
                'format': 'netcdf',
                'product_type': self.product_type,
                'variable': self.variable,
                'year': self.year,
                'month': self.month,
                'time': '00:00',
                'area': self.area,
            }

    def get_varname(self, ds):
        """get main variable name from netCDF.Dataset
        """
        variables = [v for v in ds.variables if ds[v].dimensions == ('time', 'latitude', 'longitude')]
        assert len(variables) == 1, f'Expected one variable matching (time, latitude, longitude) dimensions, found {len(variables)}.\nVariables: {ds.variables}'
        return variables[0]

    def extract_timeseries(self, lon, lat):
        if not os.path.exists(self.downloaded_file):
            self.download()
        with nc.Dataset(self.downloaded_file) as ds:
            variable = self.get_varname(ds)
            time, units = convert_time(ds)
        region = xr.open_dataset(self.downloaded_file)[variable]
        # latitude is in reverse order for this dataset
        interpolator = RegularGridInterpolator((region.longitude, region.latitude[::-1]), region.values.T[:, ::-1])
        timeseries = interpolator(np.array((lon, lat)), method='linear').squeeze()
        series = pd.Series(timeseries, index=time, name=self.variable)
        series.index.name = units
        return series


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
    if lon > 180: lon -= 360
    lons = np.arange(-180, 180, dx)
    lats = np.arange(-90, 90, dy)
    j = np.searchsorted(lons, lon)
    i = np.searchsorted(lats, lat)
    area = lats[i], lons[j-1], lats[i-1], lons[j]
    return np.array(area).tolist() # convert numpy data type to json-compatible python objects


locations = yaml.load(open('locations.yml'))
assets = yaml.load(open('assets.yml'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('variables or asset')
    g.add_argument('--cmip5', nargs='*', default=[], help='list of CMIP5-monthly variables to download')
    g.add_argument('--era5', nargs='*', default=[], help='list of ERA5-monthly variables to download')
    g.add_argument('--asset', choices=list(assets.keys()), help='pre-defined list of variables, defined in assets.yml')

    g = parser.add_argument_group('location')
    g.add_argument('--location', choices=[loc['name'] for loc in locations], help='location name defined in locations.yml')
    g.add_argument('--lon', type=float)
    g.add_argument('--lat', type=float)

    # g = parser.add_argument_group('ERA5 area')
    # g.add_argument('--width-km', type=float, default=100, help='width of window (in km) around lon/lat (%(default)s km by default)')
    # g.add_argument('-l','--left', type=float)
    # g.add_argument('-r','--right', type=float)
    # g.add_argument('-b','--bottom', type=float)
    # g.add_argument('-t','--top', type=float)

    g = parser.add_argument_group('CMIP5 control')
    g.add_argument('--model', default='ipsl_cm5a_mr')
    g.add_argument('--scenario', choices=['rcp_2_6', 'rcp_4_5', 'rcp_6_0', 'rcp_8_5'], default='rcp_8_5')

    # g = parser.add_argument_group('ERA5 control')
    # g.add_argument('--era5-start', default=2000, type=int, help='default: %(default)s')
    # g.add_argument('--era5-end', default=2019, type=int, help='default: %(default)s')

    o = parser.parse_args()

    if not (o.location or (o.lon and o.lat)):
        parser.error('please provide a location, for instance `--location Welkenraedt`, or use custom lon and lat, e.g. `--lon 5.94 --lat 50.67`')

    elif o.location:
        loc = {loc['name']: loc for loc in locations}[o.location]
        o.lon, o.lat = loc['lon'], loc['lat']

    # if o.left and o.right and o.bottom and o.top:
    #     area = o.top, o.left, o.bottom, o.right
    # else:
    #     t, l, b, r = make_area(o.lon, o.lat, o.width_km)
    #     area = o.top or t, o.left or l, o.bottom or b, o.right or r
    area = era5_tile_area(o.lon, o.lat)


    print('lon', o.lon)
    print('lat', o.lat)
    # print('area', area)

    if not o.cmip5 and not o.era5 and not o.asset:
        parser.error('please provide ERA5 or CMIP5 variables, for example: `--era5 2m_temperature` or `--cmip5 2m_temperature`, or a registered asset, e.g. `--asset energy`')

    elif o.asset:
        # use variables pre-defined by asset
        print(assets)
        asset = assets[o.asset]
        for v in asset.get('cmip5', []):
            o.cmip5.append(v)
        for v in asset.get('era5', []):
            o.era5.append(v)

    print('ERA5 variables', o.era5)
    print('CMIP5 variables', o.cmip5)


    variables = []
    for name in o.cmip5:
        variables.append(CMIP5(name, o.model, o.scenario, '200601-210012'))

    for name in o.era5:
        variables.append(ERA5(name, area=area))

    # download and convert to csv
    for v in variables:
        # series = v.extract_timeseries(o.lon, o.lat)
        series = v.save_csv(o.lon, o.lat)

if __name__ == '__main__':
    main()