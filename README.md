# iisd-cdstoolbox

Code related to offline processing of CDS Toolbox and CDS API data 

## Installation steps

- Install Python 3 (for instance anaconda python)
- Install the CDS API key: https://cds.climate.copernicus.eu/api-how-to
- Install the CDS API client: `pip install cdsapi`
- Install other dependencies: `pip install -r requirements.txt`
- Optional dependency for coastlines on plots: `conda install -c conda-forge cartopy` or see [docs](https://scitools.org.uk/cartopy/docs/latest/installing.html)

**Troubleshooting**

If the `pip install` command fails, and you have python anaconda installed, try the following:

	conda install --file requirements.txt

If that fails too, you may need to go through the dependencies in `requirements.txt` one by one and try either `pip install` or `conda install` or other methods specific to that dependency.

In the examples that follow, if you have both python2 and python3 installed, you might need to replace `python` with `python3`.

## cds api

Examples of use:

    python download_variables.py --era5 2m_temperature 10m_wind_speed --lon 5.94 --lat 50.67
    python download_variables.py --location Welkenraedt --era5 2m_temperature --cmip5 2m_temperature --view-timeseries
    python download_variables.py --location Welkenraedt --era5 2m_temperature --cmip5 2m_temperature --view-all
    python download_variables.py --location Welkenraedt --cmip5 2m_temperature --view-all --width 2000
    python download_variables.py --help
    
The variables are downloaded under `download`, and timeseries are extracted as csv file under `csv`.

(work in progress)

Planned features:

- more metadata to csv files (like units)
- `--variable temperature` (or precip etc) flag that is defined across datasets, and comes with some post-processing (e.g. temperature in degrees Celsius)
- `--asset energy` parameter to download a predefined set of variables, and save these to one CSV file
- bias-correction

## netcdf to csv

Convert netcdf files downloaded from the CDS Toolbox into csv files:

    python netcdf_to_csv.py data/*nc

Help:

    python netcdf_to_csv.py --help