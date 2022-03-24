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
import cftime

#from netcdf_to_csv import convert_time

start_year = 1979
time_units = f'days since {start_year}-01-01'

def convert_time_units_series(index, years=False):
    """convert pandas index to common units
    """
    # make sure we use the same start year
    if index.name != time_units:
        dates = cftime.num2date(index, index.name)
        index = pd.Index(cftime.date2num(dates, time_units), name=time_units)
    if years:
        index = index / 365.25 + start_year
        index.name = 'years'
    return index


class Indicator:
    """A class to compose CDS datasets into one custom Indicator (e.g. u and v components of wind into wind magnitude)
    """
    def __init__(self, name, units, description, datasets, compose=None, transform=None):
        self.name = name
        self.units = units
        self.description = description
        self.datasets = datasets
        assert len(datasets) > 0
        if compose is None and len(datasets) == 1:
            compose = lambda x:x  # identity: do not chamge anything
        self.compose = compose
        self.transform = transform

    def download(self):
        for dataset in self.datasets:
            dataset.download()

    def load_timeseries(self, lon, lat, **kwargs):
        values = [dataset.load_timeseries(lon, lat, **kwargs) for dataset in self.datasets]
        result = self.compose(*values)
        if self.transform:
            result = self.transform(result)
        result.name = f'{self.name} ({self.units})' if self.units else self.name
        return result

    def load_cube(self, *args, **kwargs):
        values = [dataset.load_cube(*args, **kwargs) for dataset in self.datasets]
        result = self.compose(*values)
        if self.transform:
            result = self.transform(result)
        result.attrs['units'] = self.units
        return result

    def __repr__(self):
        return f'{self.name} ({len(self.datasets)})'



class Dataset:

    lon0 = None    # works with either [0, 360] or [-180, 180] syntax

    def __init__(self, dataset, params, downloaded_file, transform=None, units=None, frequency=None, sub_requests=None):
        self.dataset = dataset
        self.params = params
        self.downloaded_file = downloaded_file
        self.transform = transform
        self.units = units
        self.frequency = frequency
        self.sub_requests = sub_requests or []

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

    def download(self, timeout=60*50, overwrite=False, wait_until_complete=True):
        if self.sub_requests:
            results = [dataset.download(timeout, overwrite, wait_until_complete=False) for dataset in self.sub_requests]
            for dataset, res in zip(self.sub_requests, results):
                if res is not None:
                    res.download(dataset.downloaded_file)
            return

        if os.path.exists(self.downloaded_file) and not overwrite:
            return

        c = cdsapi.Client(timeout=timeout+2, wait_until_complete=wait_until_complete)

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
            self.params)

        if wait_until_complete:
            res.download(self.downloaded_file)

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


    def _fixed_longitude(self, lon):
        """transform [-180, 180] into [0, 360]"""
        if self.lon0 is None:
            return lon

        if self.lon0 == 0:
            if lon < 0:
                return lon + 360

        if self.lon0 == -180:
            if lon >= 180:
                return lon - 360

        return lon


    def _extract_timeseries(self, f, lon, lat, transform=True):
        if not os.path.exists(self.downloaded_file):
            self.download()
        with nc.Dataset(f) as ds:
            variable = self.get_varname(ds)
            # convert to YYYY-MM-DD dates using netCDF-specific units and calendar
            time0 = cftime.num2date(ds["time"][:], ds["time"].units, calendar=ds["time"].calendar)
            # convert back to days using the standard calendar and custom units
            time = pd.Index(cftime.date2num(time0, time_units), name=time_units)

        region = xr.open_dataset(f)[variable]
        londim = getattr(region, self.lon)
        latdim = getattr(region, self.lat)
        # is latitude is in reverse order for this dataset ?
        if latdim[1] < latdim[0]:
            interpolator = RegularGridInterpolator((londim, latdim[::-1]), region.values.T[:, ::-1])
        else:
            interpolator = RegularGridInterpolator((londim, latdim), region.values.T)
        try:
            timeseries = interpolator(np.array((lon, lat)), method='linear').squeeze()
        except ValueError as error:
            print("ERROR: requested lon, lat:", (lon, lat), "but bounds are londim", londim[[0, -1]].values, " and latdim", latdim[[0, -1]].values)
            print("ERROR: This is likely an issue with the model grid as provided by Copernicus CDS.")
            print("ERROR: Recommended solution: request a nearby location within the bounds or try other climate models.")
            raise

        units = region.units # file
        series = pd.Series(timeseries, index=time, name=f'{self.variable} ({units})')

        if transform:
            series = self._transform_units(series)

        return series


    def _nc_area(self, f):
        with nc.Dataset(f) as ds:
            londim = ds[self.lon]
            latdim = ds[self.lat]
            l, r = londim[0].tolist(), londim[len(londim)-1].tolist()
            b, t = latdim[0].tolist(), latdim[len(latdim)-1].tolist()

            # In an instance (cesm2, ssp585, tas) the negative longitude were NaN (but the values are fine)
            if l is None or r is None:
                print("!! Warning: NaNs were found in lon coordinate => attempt to FIX it", f)
                londim = londim[:].filled()
                indices = np.arange(len(londim))
                valid_lon = londim[np.isfinite(londim)]
                valid_idx = indices[np.isfinite(londim)]
                dlon = valid_lon[1] - valid_lon[0]
                l = valid_lon[0] - (valid_idx[0] - 0)*dlon
                r = valid_lon[-1] + (indices[-1] - valid_idx[-1])*dlon
                londim = np.arange(l, r+dlon, dlon)
                assert len(londim) == len(ds[self.lon])

            # account for grid step
            dlon = np.abs(londim[1] - londim[0])
            l -= dlon/2
            r += dlon/2
            # dlat = np.abs(latdim[1] - latdim[0])
            # for some reason lat_bnds seems to be such that the problem does not apply

        if b > t:
            b, t = t, b

        return t, l, b, r


    def _within_ncfile(self, f, lon, lat):
        ' check if lon, lat point is within the netCDFfile'
        try:
            t, l, b, r = self._nc_area(f)
        except:
            print("!! Error while extracting netCDF area", f)
            raise
        # print('within_ncfile debug', f, (l, r, b, t), lon, lat)
        if lon < l: return False
        if lon > r: return False
        if lat < b: return False
        if lat > t: return False
        return True


    def extract_timeseries(self, lon, lat, transform=True):
        lon = self._fixed_longitude(lon)
        try:
            ncfiles = self.get_ncfiles()
        except:
            print("!! Failed getting netCDF files for", self.downloaded_file)
            raise
        files = [f for f in ncfiles if self._within_ncfile(f, lon, lat)]
        assert files, f'no file contains (lon: {lon}, lat: {lat}): {self.get_ncfiles()}'
        return pd.concat([self._extract_timeseries(f, lon, lat, transform) for f in files])


    def _transform_units(self, series):
        if self.transform:
            series = self.transform(series)
            series.name = f'{self.variable} ({self.units})'

        return series

    def timeseries_file(self, lon, lat):
        lon = self._fixed_longitude(lon)
        base, ext = os.path.splitext(self.downloaded_file)
        return base + f'_{lat}N_{lon}E.csv'

    def load_timeseries(self, lon, lat, overwrite=False):
        '''extract timeseries but buffer file...'''
        lon = self._fixed_longitude(lon)
        fname = self.timeseries_file(lon, lat)
        if not os.path.exists(fname) or overwrite:
            timeseries = self.extract_timeseries(lon, lat, transform=False)
            save_csv(timeseries, fname)

        timeseries = load_csv(fname)
        timeseries = self._transform_units(timeseries)

        return timeseries[timeseries.index >= 0]  # only load data after 1979


    def load_cube(self, time=None, area=None, roll=False):
        files = self.get_ncfiles()
        with nc.Dataset(files[0]) as ds:
            variable = self.get_varname(ds)

        if len(files) == 1:
            cube = xr.open_dataset(files[0])[variable]
        else:
            cube = xr.open_mfdataset(files, combine='by_coords')[variable]

        if time is not None:
            cube = cube.sel(time=time)

        # rename coordinates
        cube = cube.rename({self.lon: 'lon', self.lat: 'lat'})

        if roll:
            cube = roll_longitude(cube)

        lat = cube.lat.values

        # we want to deal with increasing lat
        if lat[1] < lat[0]:
            cube = cube.isel({'lat':slice(None, None, -1)})
            lat = cube.coords['lat'].values
            assert lat[1] > lat[0]

        if area is not None:
            cube = select_area(cube, area)

        if self.transform:
            try:
                cube = self.transform(cube)
            except Exception as error:
                logging.error(f'{type(self)}, {self.variable}: transform failed: {error}. Skip.')
                raise

        if self.units:
            cube.attrs['units'] = self.units  # enforce user-defined units if defined

        return cube


def roll_longitude(cube):
    ''' [0, 360] into [-180, 180] '''
    lon = cube.lon.values
    if lon[-1] > 180:
        # [0, 360] into [-180, 180]
        lon = np.where(lon <= 180, lon, lon - 360)
    else:
        # [-180, 180] into [0, 360]
        lon = np.where(lon >= 0, lon, lon + 360)
    return cube.assign_coords(lon=lon).roll(lon=lon.size//2, roll_coords=True)


def select_area(cube, area):
    t, l, b, r = area
    lat = cube.lat.values
    lon = cube.lon.values
    ilon = ((lon >= l) & (lon <= r))
    ilat = ((lat <= t) & (lat >= b))
    return cube.isel({'lon':ilon, 'lat':ilat})


def cube_area(cube, extent=False):
    assert 'lon' in cube.coords and 'lat' in cube.coords, 'cannot calculate cube_area without lon and lat coordinates'
    lat = cube.lat.values
    lon = cube.lon.values
    if (lon.size < 2) or (lat.size < 2):
        raise ValueError('region area is too small: point-wise map')
    # add extent as an attribute, for further plotting
    l = lon[0] - (lon[1]-lon[0])/2 # add half a grid cell
    r = lon[-1] + (lon[-1]-lon[-2])/2
    b = lat[0] - (lat[1]-lat[0])/2
    t = lat[-1] + (lat[-1]-lat[-2])/2
    if extent:
        return np.array((l, r, b, t)).tolist() # for imshow...
    else:
        return np.array((t, l, b, r)).tolist()


class CMIP5(Dataset):

    lon = 'lon'
    lat = 'lat'
    lon0 = 0

    def __init__(self, variable, model, experiment, period=None, ensemble=None, historical=None, frequency=None, **kwargs):
        if ensemble is None:
            ensemble = 'r1i1p1'

        if frequency is None:
            frequency = 'monthly'
        self.frequency = frequency

        if frequency == 'daily':
            from cmip5 import get_daily_periods
            dataset = 'projections-cmip5-daily-single-levels'
            if period is None:
                period = get_daily_periods(model, experiment)
        else:
            dataset = 'projections-cmip5-monthly-single-levels'
            if period is None:
                period = ['185001-200512'] if experiment == 'historical' else ['200601-210012']

        if type(period) is str:
            period = [period]
        periodstamp = period[0].split('-')[0] + '-' + period[-1].split('-')[-1]

        folder = os.path.join('download', dataset)
        name = f'{variable}-{model}-{experiment}-{periodstamp}-{ensemble}'

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

        self.historical = historical


    def load_timeseries(self, *args, **kwargs):
        series = super().load_timeseries(*args, **kwargs)
        if self.historical is None:
            return series
        historical = self.historical.load_timeseries(*args, **kwargs)
        return pd.concat([historical, series]) #TODO: check name and units


    def get_ncfiles(self):
        # download zip file
        if not os.path.exists(self.downloaded_file):
            raise FileNotFoundError(f"Not found: {self.downloaded_file}")
            # self.download()

        # extract all files if necessary
        with zipfile.ZipFile(self.downloaded_file, 'r') as zipObj:
            listOfiles = sorted(zipObj.namelist())

            if not os.path.exists(os.path.join(self.folder, listOfiles[0])):
                print(f'{self.downloaded_file} : extracting all files...')
                zipObj.extractall(path=self.folder)

        return [os.path.join(self.folder, name) for name in listOfiles]


class CMIP6(Dataset):

    lon = 'lon'
    lat = 'lat'
    lon0 = 0

    def __init__(self, variable, model, experiment, date=None, historical=None, frequency=None, period=None, area=None, **kwargs):
        # if ensemble is None:
        #     ensemble = 'r1i1p1'

        if frequency is None:
            frequency = 'monthly'
        self.frequency = frequency

        if model in ["mcm_ua_1_0"]:
            self.lon = "longitude"
            self.lat = "latitude"

        dataset = 'projections-cmip6'

        if period:
            raise DeprecationError("`period` is not an argument of the CMIP6 class. `date` is determined automatically")

        date = ['1900-12-01/2014-12-31'] if experiment == 'historical' else ['2015-01-01/2100-12-31']

        if date is str:
            date = [date]
        datestamp = date[0].split("/")[0].replace("-","") + '-' + date[-1].split("/")[1].replace("-","")


        folder = os.path.join('download', dataset)
        name = f'{variable}-{model}-{experiment}-{datestamp}'

        if area is not None:
            area = np.array(area).tolist() # be sure it is json serializable
            name += "_" + f'{area[0]}N-{area[1]}E-{area[2]}N-{area[3]}E'

        downloaded_file = os.path.join(folder, name+'.zip')

        super().__init__(dataset,
            {
                'temporal_resolution': frequency,
                'experiment': experiment,
                'level': 'single_levels',
                'variable': variable,
                'model': model,
                'date': date,
                'area': area,
                # 'ensemble_member': ensemble,
                'format': 'zip',
            }, downloaded_file, **kwargs)

        # initialize an `historical` attribute
        if historical is True:
            historical = CMIP6(variable, model, "historical", date, frequency=frequency, period=period, area=area, **kwargs)
        elif historical is False:
            historical = None

        self.historical = historical


    def load_timeseries(self, *args, **kwargs):
        series = super().load_timeseries(*args, **kwargs)

        if self.historical is None:
            return series

        historical = self.historical.load_timeseries(*args, **kwargs)

        return pd.concat([historical, series]) #TODO: check name and units


    def get_ncfiles(self):
        # download zip file
        if not os.path.exists(self.downloaded_file):
            raise FileNotFoundError(f"Not found: {self.downloaded_file}")
            # self.download()

        zipfolder = os.path.splitext(self.downloaded_file)[0]

        # extract all files if necessary
        with zipfile.ZipFile(self.downloaded_file, 'r') as zipObj:
            listOfiles = sorted([f for f in zipObj.namelist() if f.endswith(".nc")])
            os.makedirs(zipfolder, exist_ok=True)

            if not os.path.exists(os.path.join(zipfolder, listOfiles[0])):
                print('Extracting all files...')
                zipObj.extractall(path=zipfolder)

        return [os.path.join(zipfolder, name) for name in listOfiles]

    def download(self):
        super().download()

        if self.historical is not None:
            self.historical.download()



class ERA5(Dataset):

    lon = 'longitude'
    lat = 'latitude'


    def __init__(self, variable, year=None, area=None, frequency=None, split_year=None, **kwargs):
        """
        """
        if area is None:
            area = [90, -180, -90, 180]
        else:
            area = np.array(area).tolist() # be sure it is json serializable
        if year is None:
            year = list(range(1979, 2019+1))  # multiple year OK

        if frequency is None:
            frequency = 'monthly'

        if frequency == 'hourly':
            dataset = 'reanalysis-era5-single-levels'
            product_type = 'reanalysis'
            split_year = True if split_year is not False else False # otherwise item limit is exceeded
        elif frequency == 'monthly':
            dataset = 'reanalysis-era5-single-levels-monthly-means'
            product_type = 'monthly_averaged_reanalysis'
        else:
            raise ValueError(f'expected monthly or hourly frequency, got: {frequency}')

        self.frequency = frequency

        folder = os.path.join('download', dataset, product_type)
        year0, yearf = year[0], year[-1]
        name = f'{variable}_{year0}-{yearf}_{area[0]}-{area[1]}-{area[2]}-{area[3]}'
        downloaded_file = os.path.join(folder, name+'.nc')

        sub_requests = []

        if split_year:
            sub_requests = [ERA5(variable, [y], area, frequency=frequency, split_year=False, **kwargs) for y in year]

        params = {
            'format': 'netcdf',
            'product_type': product_type,
            'variable': variable,
            'year': year,
            'month': list(range(1, 12+1)),
            'time': '00:00',
            'area': area,
        }

        if frequency == 'hourly':
            params['day'] = list(range(1, 31+1))
            params['time'] = list(range(0, 23+1))

        super().__init__(dataset, params, downloaded_file, sub_requests=sub_requests, **kwargs)

    def get_ncfiles(self):
        if not os.path.exists(self.downloaded_file):
            raise FileNotFoundError(f"Not found: {self.downloaded_file}")
            # self.download()

        if self.sub_requests:
            ncfiles = []
            for v in self.sub_requests:
                ncfiles.extend(v.get_ncfiles())
            return ncfiles

        return [self.downloaded_file]

# class ERA5hourly(ERA5):
#     pass

def make_area(lon, lat, w, precision=1):
    " return `area` keyword top left bottom right for lon/lat and width_km (one digit after 0 by default)"
    earth_radius = 6371
    latw = np.rad2deg(w/earth_radius)
    disk_radius = earth_radius * np.cos(np.deg2rad(lat))
    lonw = np.rad2deg(w/disk_radius)
    return np.round(lat+latw, precision), np.round(lon-lonw, precision), np.round(lat-latw, precision), np.round(lon+lonw, precision)


def load_csv(fname):
    series = pd.read_csv(fname, index_col=0, comment='#').squeeze()
    series.index = convert_time_units_series(series.index) # just in case units are different
    return series

def save_csv(series, fname):
    series.to_csv(fname)


def monthly_climatology(dates, values, interval=None):
    # select interval
    date_val = zip(dates, values)

    if interval:
        y1, y2 = interval
        date_val = [(date, val) for date, val in date_val if date.year >= y1 and date.year <= y2]

    monthkey = lambda xy: xy[0].month
    monthly_clim = [np.mean([val for date,val in g]) for month, g in itertools.groupby(sorted(date_val, key=monthkey), key=monthkey)]
    assert len(monthly_clim) == 12
    return np.array(monthly_clim)


def yearly_climatology(dates, values, interval=None):
    # select interval
    date_val = zip(dates, values)

    if interval:
        y1, y2 = interval
        date_val = [(date, val) for date, val in date_val if date.year >= y1 and date.year <= y2]

    return np.mean([val for date, val in date_val])


def _correct_bias(values, clim, target, method):
    if method == 'offset':
        return values + (target - clim)
    elif method == 'percent':
        if clim == 0:
            logging.warning(f'zero-value in % bias correction ({method})')
            return np.nan
        return (values - clim) * (target / clim) + target
    elif method == 'scale':
        if clim == 0:
            logging.warning(f'zero-value in % bias correction ({method})')
            return np.nan
        return values * (target / clim)
    else:
        raise NotImplementedError(method)


def correct_monthly_bias(series, era5, interval, method):
    """ correct for each month
    """
    dates = nc.num2date(series.index, time_units)
    era5_dates = nc.num2date(era5.index, time_units)

    era5_clim = monthly_climatology(era5_dates, era5.values, interval)
    cmip5_clim = monthly_climatology(dates, series.values, interval)
    # delta = era5_clim - cmip5_clim

    print(f'Applying "{method}" bias correction for {series.name}.')
    print(' - yearly bias prior correction:', np.mean(cmip5_clim - era5_clim))

    # apply monthly anomaly
    unbiased = series.values.copy()
    for i, date in enumerate(dates):
        unbiased[i] = _correct_bias(unbiased[i], cmip5_clim[date.month -1], era5_clim[date.month -1], method)

    print(' - yearly bias after correction:', np.mean(monthly_climatology(dates, unbiased, interval) - era5_clim))

    return pd.Series(unbiased, index=series.index, name=series.name)


def correct_yearly_bias(series, era5, interval, method):
    """ correct for each month
    """
    dates = nc.num2date(series.index, time_units)
    era5_dates = nc.num2date(era5.index, time_units)

    era5_clim = yearly_climatology(era5_dates, era5.values, interval)
    cmip5_clim = yearly_climatology(dates, series.values, interval)

    print(f'Applying "{method}" bias correction for {series.name}.')
    print(' - yearly bias prior correction:', cmip5_clim - era5_clim)

    unbiased = _correct_bias(series.values, cmip5_clim, era5_clim, method)

    print(' - yearly bias after correction:', yearly_climatology(dates, unbiased, interval) - era5_clim)

    return pd.Series(unbiased, index=series.index, name=series.name)
