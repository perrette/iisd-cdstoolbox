"""Get information from https://cp-availability.ceda.ac.uk/data-availability
"""
import os
import json
import yaml
import glob

root = '../'

indicators = yaml.safe_load(open(root+'indicators.yml'))
assets = yaml.safe_load(open(root+'assets.yml'))
cmip5 = yaml.safe_load(open(root+'cmip5.yml'))

all_models = cmip5['models']

per_indicator = {}
per_asset = {}

for v in indicators:
    name = v['name']
    per_indicator[name] = models = []
    # cmip5_name = v.get('cmip5', {}).get('name', name)
    d = json.load(open(name+'.json'))
    for result in d['results']:
        models.append( result['model'].lower().replace('-','_') )
    #print(name, models)

#for name in assets:
for name in ['energy']:
    #models = set(all_models)
    models = None
    for v in assets[name]:
        print(v, per_indicator[v])
        if models is None:
            models = set(per_indicator[v])
        else:
            models = models.intersection(set(per_indicator[v]))
    per_asset[name] = sorted(models)
    print(name, per_asset[name])
