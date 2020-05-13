# iisd-cdstoolbox

Code related to offline processing of CDS Toolbox and CDS API data for the C3S_428h_IISD-EU project.

## How does this code relate to the CDS API ?

This code builds on the powerful CDS API but focuses on local impact analysis specific for the C3S_428h_IISD-EU project. It makes it easier to retrieve a timeseries for a specific location or region, and save the result to a CSV file (a simpler format than netCDF for most climate adaptation practitioners). The next step will be to combine variables across multiple datasets, and aggregate them into asset classes (such as all energy-related variables) and perform actions such as bias correction (use of ERA5 and CMIP5).

## Download this code

The easy way is to download the zipped archive:
- latest (development): https://github.com/perrette/iisd-cdstoolbox/archive/master.zip
- or check stable releases with description of changes: https://github.com/perrette/iisd-cdstoolbox/releases (see assets at the bottom of each release to download a zip version)

The hacky way is to use git (only useful during development, for frequent updates, to avoid having to download and extract the archive every time):
- First time: `git clone https://github.com/perrette/iisd-cdstoolbox.git`
- Subsequent updates: `git pull` from inside the repository 

## Installation steps

- Download the code (see above) and inside the folder.
- Install Python 3, ideally Anaconda Python which comes with pre-installed packages
- Install the CDS API key: https://cds.climate.copernicus.eu/api-how-to
- Install the CDS API client: `pip install cdsapi`
- Install other dependencies: `conda install --file requirements.txt` or `pip install -r requirements.txt`
- _Optional_ dependency for coastlines on plots: `conda install -c conda-forge cartopy` or see [docs](https://scitools.org.uk/cartopy/docs/latest/installing.html)
- _Optional_ dependency: CDO (might be needed later, experimental): `conda install -c conda-forge python-cdo`


Troubleshooting:
- If install fails, you may need to go through the dependencies in `requirements.txt` one by one and try either `pip install` or `conda install` or other methods specific to that dependency.
- In the examples that follow, if you have both python2 and python3 installed, you might need to replace `python` with `python3`.

## cds api

(work in progress)

**Examples of use**:

Download ERA5 `2m_temperature` (monthly, single level):

    python download_variables.py --era5 2m_temperature 10m_wind_speed --lon 5.94 --lat 50.67

Use pre-defined location instead of `--lon` and `--lat`, and show a plot of downloaded time series for ERA5 and CMIP5:

    python download_variables.py --location Welkenraedt --era5 2m_temperature --cmip5 2m_temperature --view-timeseries
    
Show a map + time series of the downloaded data:
    
    python download_variables.py --location Welkenraedt --era5 2m_temperature --cmip5 2m_temperature --view-all
    
Enlarge the view with 2000 km, for CMIP5 (but see known issues below):
    
    python download_variables.py --location Welkenraedt --cmip5 2m_temperature --view-all --width 2000
    
Full documentation:

    python download_variables.py --help
    

**Implemented Features**:

- Download any ERA5 and CMIP5 variable directly to `csv` file, for any lon/lat location
- The raw CDS API files are downloaded under `download`, and timeseries are extracted as csv file under `csv`.
- Favourite locations can be defined in [locations.yml](locations.yml) and called with `--location` parameter, to avoid having to indicate `--lon`, `--lat` each time.
- ERA5 is downloaded for the 2000-2019 period, CMIP5 for 2006-2100. 
- Default CMIP5 scenario is `rcp_8_5` (RCP 8.5) and default CMIP5 model is `ipsl_cm5a_mr`. 
- Any other model or and any other future scenario can be specified via `--model` and `--scenario` parameters.
- CMIP5 data are downloaded for the whole globe (in zipped format), about 70Mb per variable and per model for the 2006-2100 period
- ERA5 data are downloaded in predefined "tiles" (rectangles) of 5 degrees latitude and 10 degrees longitude, for easier re-use in nearby locations (no need to download the full file again)
- Visualize results via `--view-timeseries`, `--view-region` or `--view-area` parameters (experimental).
- Possibility to specify a different region via `--width` (experimental, with issues)


**Planned features**:

- more metadata to csv files (like units)
- `--variable temperature` (or precip etc) parameter that will be defined and harmonized across datasets, and comes with some post-processing (e.g. temperature in degrees Celsius)
- `--asset energy` parameter to download a predefined set of variables, and save these to one CSV file
- bias-correction by combining CMIP5 and ERA5
- technical: harmonize downloaded netCDF files (e.g. dimensions are now named `lon` in CMIP5 and `longitude` in ERA5. Having one name will make simpler code)

**Features under consideration**:

- add support for more datasets ?


**Known issues**:

- Problems for longitudes outside the [0, 180] range: will be solved in a future release.
- Plotting areas for large regions that cross the longitude range above cause problems, for the reason outlined above.


## netcdf to csv

Convert netcdf files downloaded from the CDS Toolbox into csv files:

    python netcdf_to_csv.py data/*nc

Help:

    python netcdf_to_csv.py --help
