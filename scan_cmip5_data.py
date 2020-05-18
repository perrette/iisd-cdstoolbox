import yaml
import json
from download_indicators import CMIP5

cmip5 = yaml.safe_load(open('cmip5.yml'))

# for experiment in cmip5['experiment']:
listing = []

try:
    for experiment in 'rcp_8_5':
        for model in cmip5['models']:
            for variable in cmip5['variables']:
                request = CMIP5(variable, model, experiment, '2006-2100')
                try:
                    request.download()
                    listing.append((variable, model, experiment, True))
                except:
                    listing.append((variable, model, experiment, False))
finally:
    pass


json.dump({'listing':listing}, open('cmip5_listing.json','w'))