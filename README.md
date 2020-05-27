# iisd-cdstoolbox

Code related to offline processing of CDS Toolbox and CDS API data for the C3S_428h_IISD-EU project.

## How does this code relate to the CDS API ?

This code builds on the powerful CDS API but focuses on local impact analysis specific for the C3S_428h_IISD-EU project. It makes it easier to retrieve a timeseries for a specific location or region, and save the result to a CSV file (a simpler format than netCDF for most climate adaptation practitioners). Additionally, the code combines variables across multiple datasets, aggregate them into asset classes (such as all energy-related variables) and perform actions such as bias correction (use of ERA5 and CMIP5).

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
- Install other [dependencies](requirements.txt): `conda install --file requirements.txt` or `pip install -r requirements.txt`
- _Optional_ dependency for coastlines on plots: `conda install -c conda-forge cartopy` or see [docs](https://scitools.org.uk/cartopy/docs/latest/installing.html)
- _Optional_ dependency: CDO (might be needed later, experimental): `conda install -c conda-forge python-cdo`


Troubleshooting:
- If install fails, you may need to go through the dependencies in `requirements.txt` one by one and try either `pip install` or `conda install` or other methods specific to that dependency.
- In the examples that follow, if you have both python2 and python3 installed, you might need to replace `python` with `python3`.

## cds api

Download indicators associated with one asset class.

**Examples of use**:

    python download.py --asset energy --location Welkenraedt 
    
The corresponding csv timeseries will be stored in `indicators/welkenraedt/energy`. Note that raw downloaded data from the CDS API (regional tiles in netcdf format, and csv for the required lon/lat, without any correction) are stored under `download/` and can be re-used across multiple indicators.

The `indicators` folder is organized by location, asset class, simulation set and indicator name. The aim is to provide multiple sets for Savi simulation. For instance, `era5` for past simulations, and various `cmip5` versions for future simulations, that may vary with model and experiment. For instance the above command creates the folder structure (here a subset of all variables is shown):

	indicators/
	  welkenraedt/
	    energy/
		  era5/
			2m_temperature.csv
			precipitation.csv
			...
		  cmip5-ipsl_cm5a_mr-rcp_8_5/
			2m_temperature.csv
			precipitation.csv
			...
		  ...

with two simulation sets `era5` and `cmip5-ipsl_cm5a_mr-rcp_8_5`. It is possible to specify other models and experiment via `--model`  and `--experiment` parameters, to add futher simulation sets and thus test how the choice of climate models and experiment affect the result of Savi simulations.

Compared to raw CDS API, some variables are renamed and scaled so that units match and are the same across simulation sets.
For instance, temperature was adjusted from Kelvin to degree Celsius, and precipitation was renamed and units-adjusted into mm per month from original (mean_total_precipitation_rate (mm/s) in ERA5, and mean_precipitation_flux (mm/s) in CMIP5). Additionally, CMIP5 data is corrected so that climatological mean matches with ERA5 data (climatology computed over 1979-2019 by default). 

Additionally to the files shown in the example folder listing above, figures can also be created for rapid control of the data, either for interactive viewing (`--view-timeseries` and `--view-region`) or or saved as PNG files (`--png-timeseries` and `--png-region`), e.g.

	python download.py --asset energy --location Welkenraedt --png-timeseries --png-region

Single indicators can be downloaded via:

	python download.py --indicator 2m_temperature --location Welkenraedt

The choices available for `--indicator` , `--asset` and `--location` area defined in the following configuration files, respectively:

- controls which indicators are available, how they are renamed and unit-adjusted: [indicators.yml](indicators.yml) (see [sub-section](#indicator-definition) below)
- controls the indicator list in each asset class: [assets.yml](assets.yml)
- controls the list of locations available: [locations.yml](locations.yml)

Full documentation, including fine-grained controls, is provided in the command-line help:

    python download.py --help


Visit the CDS Datasets download pages, for more information about available variables, models and scenarios:
- ERA5: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
- CMIP5: `https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip5-monthly-single-levels?tab=form`
In particular, clicking on "Show API request" provides information about spelling of the parameters, e.g. that "2m temperature" is spelled `2m_temperature` and "RCP 8.5" is spelled `rcp_8_5`.



### Indicator definition

This section is intended for users who wish to extend the list of indicators currently defined in [indicators.yml](indicators.yml).
It can be safely ignored for users who are only interested in using the existing indicators.

Let's see how `10m_wind_speed` is defined:

	- name: 10m_wind_speed
	  units: m / s
	  description: Wind speed magnitude at 10 m

The fields `name` and `units` define the indicator. Description is optional, just to provide some context.
It is possible to provide `scale` and `offset` fieds to correct the data as `(data + offset) * scale`.
Here for `2m temperature`:

	- name: 2m_temperature
	  units: degrees Celsius
	  description: 2-m air temperature
      offset: -273.15  # Kelvin to degrees C

`#` denotes a comment to provide some context. 
Some indicators have different names in ERA5 and CMIP5, and possibly different units. 
That can be dealt with by providing `era5` and `cmip5` fields, which have precedence over the top-level fields. 
Here the `evaporation` definition:

	- name: evaporation
	  units: mm per month
	  era5:
	    name: mean_evaporation_rate  # different name in ERA5
	    scale: -2592000  # change sign and convert from mm/s to mm / month
	  cmip5:
	    scale: 2592000  # mm/s to mm / month

In that case both scaling and name depend on the dataset. In CMIP5 which variable name is identical to our indicator name, the `name` field can be omitted.
In ERA5, evaporation is negative (downwards fluxes are counted positively), whereas it is counted positively in ERA5.

Indicators composed of several CDS variables can be defined via `compose` and `expression` fields.
Let's look at `100m_wind_speed`:

	- name: 100m_wind_speed
	  units: m / s
	  description: Wind speed magnitude at 100 m
	  era5:
	    compose:
	      - 100m_u_component_of_wind
	      - 100m_v_component_of_wind
	    expression: (_100m_u_component_of_wind**2 + _100m_v_component_of_wind**2)**0.5
	  cmip5:
	    name: 10m_wind_speed
	    scale: 1.6  # average scaling from 10m to 100m, based on one test location (approximate!)

In ERA5, vector components of 100m wind speed are provided. 
Our indicator is therefore a composition of these two variables, defined by the `expression` field, which is evaluated as a python expression.
Note that variables that start with a digit are not licit in python and must be prefixed with an underscore `_` in the `expression` field (only there).

For complex expressions, it is possible to provide a `mapping` field to store intermediate variables, for readability. This is used for the `relative_humidity` indicator:

	- name: relative_humidity
	  units: '%'
	  era5:
	    compose:
	      - 2m_temperature
	      - 2m_dewpoint_temperature
	    expression: 100*(exp((17.625*TD)/(243.04+TD))/exp((17.625*T)/(243.04+T)))
	    mapping: {T: _2m_temperature - 273.15, TD: _2m_dewpoint_temperature - 273.15}
	  cmip5:
	    name: near_surface_relative_humidity

where `T` and `TD` are provided as intermediary variables, to be used in `expression`.

ERA5-hourly dataset can be retrieved via `frequency: hourly` field, and subsequently aggregated to monthly indicators 
thanks to pre-defined functions `daily_max`, `daily_min`, `daily_mean`, `monthly_mean`, `yearly_mean`. For instance:

	- name: maximum_daily_temperature
	  units: degrees Celsius
	  offset: -273.15
	  cmip5:
	    name: maximum_2m_temperature_in_the_last_24_hours
	  era5:
	    name: 2m_temperature
	    frequency: hourly
	    transform: 
	      - daily_max
	      - monthly_mean

This variable is available directly for CMIP5, but not in ERA5. It is calculated from `2m_temperature` from ERA5 `hourly` dataset, and subsequently aggregated.
Note the ERA5-hourly dataset takes significantly longer to retrieve than ERA5 monthly. Consider using in combination with `--year 2000` to retrieve a single year of the ERA5 dataset.

Currently CMIP5 daily is not supported. 

## netcdf to csv

Convert netcdf timeseries files downloaded from the CDS Toolbox pages into csv files (note : this does not work for netcdf files downloaded via the cds api):

    python netcdf_to_csv.py data/*nc

Help:

    python netcdf_to_csv.py --help
