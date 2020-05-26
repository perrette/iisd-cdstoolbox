import yaml
import json
import os
import datetime
from common import CMIP5

indicators = yaml.safe_load(open('indicators.yml'))
assets = yaml.safe_load(open('assets.yml'))
cmip5 = yaml.safe_load(open('cmip5.yml'))

listing_file = 'cmip5_listing.json'

if os.path.exists(listing_file):
    listing = json.load(open(listing_file))['listing']
else:
    listing = []

def make_cmip5_listing(all_cmip5=False, experiment='rcp_8_5', asset=None, variable=None, indicator=None, lazy=False, extend=True):

    if all_cmip5:
        all_variables = cmip5['variables']
    elif asset:
        all_variables = [vdef.get('cmip5', {}).get('name', vdef.get('name')) for vdef in indicators if vdef['name'] in assets[asset]]
    elif variable:
        all_variables = [variable]
    elif indicator:
        all_variables = [vdef.get('cmip5', {}).get('name', vdef.get('name')) for vdef in indicators if vdef['name'] == indicator]
    else:
        all_variables = [vdef.get('cmip5', {}).get('name', vdef.get('name')) for vdef in indicators]


    print(all_variables)

    if not extend:
        global listing
        listing = []

    models = cmip5['models']

    # for experiment in cmip5['experiment']:
    for experiment in [experiment]:
        for model in models:
            for variable in all_variables:
                element = variable, model, experiment

                if element in listing:
                    print(element, 'checked')
                    continue

                request = CMIP5(variable, model, experiment, '200601-210012')
                try:
                    if os.path.exists(request.downloaded_file):
                        print('file already present')
                    else:
                        request.download()
                    print(element, 'added')
                    listing.append(element)
                except:
                    print(element, 'failed')
                    if lazy:
                        print('skip model', model)
                        break
                    else:
                        continue

    print('valid models for all asset', models)

    json.dump({'listing':listing}, open(listing_file,'w'))


def get_all_models():
    return cmip5['models']
    

def get_models_per_variable(cmip5_name, experiment=None):
    return sorted([model for variable, model, experiment_ in listing if variable == cmip5_name and (experiment is None or experiment == experiment_)])

def get_models_per_indicator(name, experiment=None):
    vdef = {vdef['name']:vdef for vdef in indicators}[name]
    cmip5_name = vdef.get('cmip5', {}).get('name', name)
    return get_models_per_variable(cmip5_name, experiment)


def get_models_per_asset(asset, experiment=None, verbose=True):
    all_models  = [set(get_models_per_indicator(name, experiment)) for name in assets[asset]]

    models = sorted(set.intersection(*all_models))

    if verbose:
        for name, indicator_models in zip(assets[asset], all_models):
            print('- '+name+':')
            print('  ', ' '.join(sorted(indicator_models)))

    return models


 
daily_periods = {
    'ipsl_cm5a_mr': ['19500101-19991231', '20000101-20051231', '20060101-20551231', '20560101-21001231'],
    'mpi_esm_mr': ['19700101-19791231', '19800101-19891231', '19900101-19991231', '20000101-20051231',
            '20060101-20091231', '20100101-20191231', '20200101-20291231',
            '20300101-20391231', '20400101-20491231', '20500101-20591231',
            '20600101-20691231', '20700101-20791231', '20800101-20891231',
            '20900101-21001231'], 
    'bnu_esm': ['19500101-20051231', '20060101-21001231'],
    'csiro_mk3_6_0': ['19700101-19891231', '19900101-20051231', '20060101-20251231', '20260101-20451231', '20460101-20651231',
             '20660101-20851231', '20860101-21001231'],
}

def _init_date(string):
    y, m, d = string[:4], string[4:6], string[6:]
    return datetime.date(int(y), int(m), int(d))

def get_daily_periods(model, scenario):
    """get list of valid period parameter for one model, scenario for daily frequency """
    periods = []
    for period in daily_periods[model]:
        date1, date2 = (_init_date(date) for date in period.split('-'))
        if scenario == 'historical' and date2 < datetime.date(2006,1,1):
            periods.append(period)
        if scenario.startswith('rcp') and date1 >= datetime.date(2006,1,1):
            periods.append(period)
    return periods

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--asset')
    g.add_argument('--indicator')
    g.add_argument('--variable')
    parser.add_argument('--update-listing', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--experiment', default='rcp_8_5')
    o = parser.parse_args()

    if o.update_listing:
        make_cmip5_listing(all_cmip5=False, experiment=o.experiment, asset=o.asset, variable=o.variable, indicator=o.indicator, lazy=True)

    if o.asset:
        print(' '.join(get_models_per_asset(o.asset, o.experiment, verbose=o.verbose)))
    elif o.indicator:
        print(' '.join(get_models_per_indicator(o.indicator, o.experiment)))
    elif o.variable:
        print(' '.join(get_models_per_variable(o.variable, o.experiment)))
