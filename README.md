# iisd-cdstoolbox

Code related to offline processing of CDS Toolbox and CDS API data 

## Installation steps

- Install Python 3 (for instance anaconda python)
- Install the CDS API key: https://cds.climate.copernicus.eu/api-how-to
- Install the CDS API client: `pip install cdsapi`
- Install other dependencies:

	pip install -r requirements.txt

**Troubleshooting**

If the `pip install` command fails, and you have python anaconda installed, try the following:

	conda install --file requirements.txt

If that fails too, you may need to go through the dependencies in `requirements.txt` one by one and try either `pip install` or `conda install` or other methods specific to that dependency.

In the examples that follow, if you have both python2 and python3 installed, you might need to replace `python` with `python3`.

## cds api

    python download_assset.py --asset energy --location Welkenraedt
    python download_assset.py --help
    
(work in progress)

## netcdf to csv

Convert netcdf files downloaded from the CDS Toolbox into csv files:

    python netcdf_to_csv.py data/*nc

Help:

    python netcdf_to_csv.py --help