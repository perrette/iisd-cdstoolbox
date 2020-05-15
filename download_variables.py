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


    def load_csv(self, lon, lat, fname=None):
        if fname is None:
            fname = self.csvfile(lon, lat)
        return pd.read_csv(fname, index_col=0, comment='#')


    def __repr__(self):
        return f'{type(self).__name__}({self.dataset}, {self.params}, {self.downloaded_file})'

    # def get_units(self):
    #     ' get units from netCDF file'
    #     files = self.get_ncfiles()
    #     with nc.Dataset(files[0]) as ds:
    #         varname = self.get_varname(ds)
    #         return getattr(ds[varname], 'units', '')

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

    def _extract_timeseries(self, f, lon, lat):
        if not os.path.exists(self.downloaded_file):
            self.download()
        with nc.Dataset(f) as ds:
            variable = self.get_varname(ds)
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
        series = pd.Series(timeseries, index=time, name=f'{self.variable} ({region.units})')
        series.index.name = time_units
        return series


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

        if (lon.size < 2) or (lat.size < 2):
            raise ValueError('region area is too small: point-wise map')

        # add extent as an attribute, for further plotting
        l = lon[0] - (lon[1]-lon[0])/2 # add half a grid cell
        r = lon[-1] + (lon[-1]-lon[-2])/2
        b = lat[0] - (lat[1]-lat[0])/2
        t = lat[-1] + (lat[-1]-lat[-2])/2
        map.attrs['extent'] = np.array((l, r, b, t)).tolist()
        return map


class CMIP5(Dataset):

    lon = 'lon'
    lat = 'lat'

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


    def extract_timeseries(self, lon, lat):
        files = self.get_ncfiles()
        # alltimes, allvalues = zip(*[self._extract_timeseries(f, lon, lat) for f in files])
        return pd.concat([self._extract_timeseries(f, lon, lat) for f in files])


    def extract_map(self, time=None, area=None):
        files = self.get_ncfiles()
        with nc.Dataset(files[0]) as ds:
            variable = self.get_varname(ds)
        dataarray = xr.open_mfdataset(files, combine='by_coords')[variable]
        return self._extract_map(dataarray, time, area)


class ERA5(Dataset):

    lon = 'longitude'
    lat = 'latitude'


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

    def get_ncfiles(self):
        return [self.downloaded_file]

    def extract_timeseries(self, lon, lat):
        return self._extract_timeseries(self.downloaded_file, lon, lat)


    def extract_map(self, time=None, area=None):
        files = self.get_ncfiles()
        with nc.Dataset(files[0]) as ds:
            variable = self.get_varname(ds)
        dataarray = xr.open_dataset(files[0])[variable]
        return self._extract_map(dataarray, time, area)



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


class DatasetVariable:
    """this class contains information to map raw CDS API variable from a given dataset onto the cross-dataset variable definition"""
    def __init__(self, name, dataset, scale=1, offset=0, transform=None, note='', **kwargs):
        self.name = name
        self.dataset = dataset
        self.scale = scale
        self.offset = offset
        self.transform = transform
        self.note = note
        if kwargs:
            logging.error(f'unknown field in {name}:{dataset} --> {kwargs}. Valid fields are: name, dataset, scale, offset, transform, note.')

    @classmethod
    def fromdef(cls, variable, definition):
        """from definition file"""
        if type(definition) is str:
            definition = {'name': variable.name, 'dataset': definition}
        assert isinstance(definition, dict), f'expected a dict, got {type(definition)} : {definition}'
        
        if 'name' not in definition:
            definition['name'] = variable.name

        if 'transform' in definition:
            transform = definition.pop('transform')
            logging.warning(f'{definition["name"]}: transform support is not yet implemented, use scale and offset if possible. Skipping transform.')

        return cls(**definition)

    def todef(self):
        return vars(self)

    def postprocess(self, data):
        data2 = (data + self.offset)*self.scale
        if self.transform:
            data2 = self.transform(data2)
        return data2


class Variable:
    def __init__(self, name, units, description='', datasets=None, **kwargs):
        self.name = name
        self.units = units
        self.description = description
        self.datasets = datasets or []

        if kwargs:
            logging.error(f'unknown field in for variable {name} : {kwargs}. Valid fields are: name, units, description, datasets.')

    @classmethod
    def fromdef(cls, definition):
        """from definition file"""
        dataset_definitions = definition.pop('datasets', [])

        # legacy syntax with 'era5' and 'cmip5' fields
        # --> insert these in the 'datasets' list of dataset variables
        for field in ['era5', 'cmip5']:
            if field in definition:
                logging.warning('preferred definition is as a dataset list')
                vdef = definition.pop(field)
                vdef['dataset'] = field
                dataset_definitions.append(vdef)


        variable = cls(**definition)

        for vdef in dataset_definitions:
            dataset = DatasetVariable.fromdef(variable, vdef)
            variable.datasets.append(dataset)

        return variable

    def todef(self):
        defs = vars(self).copy()
        defs['datasets'] = [dataset.todef() for dataset in defs.pop('datasets')]
        return defs

    def __repr__(self):
        return yaml.dump(self.todef(), sort_keys=False, default_flow_style=False)


locations = yaml.safe_load(open('locations.yml'))
variables_def = yaml.safe_load(open('variables.yml'))
custom_variables = [Variable.fromdef(element) for element in variables_def]
assets = yaml.safe_load(open('assets.yml'))

custom_variables_byname = {v.name:v for v in custom_variables}


def define_variable_by_name(name):
    """initialize Variable instance by name
    """
    if name not in custom_variables_byname:
        logging.warning(f'{name} is not defined in variables.yml. Standard definition assumed.')
        variable = Variable.fromdef({'name':name, 'units':'', 'datasets': ['era5', 'cmip5']})
    else:
        variable = custom_variables_byname[name]

    return variable


def define_dataset_variable(variable, dataset, **kwargs):
    if dataset == 'era5':
        year = kwargs.pop('year')
        area = kwargs.pop('area')
        return ERA5(variable.name, year, area)

    elif dataset == 'cmip5':
        model = kwargs.pop('model')
        scenario = kwargs.pop('scenario')
        period = kwargs.pop('period')
        ensemble = kwargs.pop('ensemble', None)
        return CMIP5(variable.name, model, scenario, period, ensemble=ensemble)

    else:
        raise ValueError(f'unknown dataset: {dataset}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('variables or asset')
    g.add_argument('--cmip5', nargs='*', default=[], help='list of CMIP5-monthly variables to download')
    g.add_argument('--era5', nargs='*', default=[], help='list of ERA5-monthly variables to download')
    g.add_argument('--asset', choices=list(assets.keys()), help='pre-defined list of variables, defined in assets.yml (experimental)')

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
    g.add_argument('--model', default='ipsl_cm5a_mr')
    g.add_argument('--scenario', choices=['rcp_2_6', 'rcp_4_5', 'rcp_6_0', 'rcp_8_5'], default='rcp_8_5')

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
    # print('area', area)

    if not o.cmip5 and not o.era5 and not o.asset:
        parser.error('please provide ERA5 or CMIP5 variables, for example: `--era5 2m_temperature` or `--cmip5 2m_temperature`, or a registered asset, e.g. `--asset energy`')

    elif o.asset:
        # more complex download_variables2 syntax re-used here
        # the variables defined in asset are appended to --cmip5 and --era5 flags.
        asset = assets[o.asset]  
        cvars = {v.name: v for v in custom_variables}  # pre-defined variables
        for vname in asset:
            cvar = define_variable_by_name(vname)  # Variable class 
            for dataset_variable in cvar.datasets:  
                if dataset_variable.dataset == 'era5':
                    o.era5.append(dataset_variable.name)
                elif dataset_variable.dataset == 'cmip5':
                    o.cmip5.append(dataset_variable.name)
                else:
                    logging.warning(f'dataset {dataset_variable.dataset} is not supported')

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

                if o.view_timeseries or o.png_region:
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
                        fig1.savefig(v.csvfile(o.lon, o.lat).replace('.csv', '-region.png'))
                    if not o.view_region:
                        plt.close(fig1)
                except:
                    raise
                    pass


            if o.view_timeseries or o.png_timeseries:
                ts = v.load_csv(o.lon, o.lat)
                # convert units for easier reading of graphs
                ts.index = ts.index / 365.25 + 2000
                ts.index.name = 'years since 2000-01-01'
                ts.plot(ax=ax2)
                ax2.set_title(v.dataset)

                if o.png_timeseries:
                    fig2.savefig(v.csvfile(o.lon, o.lat).replace('.csv', '.png'))
                if not o.view_timeseries:
                    plt.close(fig2)

        plt.show()


if __name__ == '__main__':
    main()
