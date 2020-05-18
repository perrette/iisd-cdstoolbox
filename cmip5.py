import yaml
import json
import os
from download import CMIP5

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
