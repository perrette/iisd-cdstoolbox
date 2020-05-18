import yaml
import json
import os
from download_indicators import CMIP5

cmip5 = yaml.safe_load(open('cmip5.yml'))
indicators = yaml.safe_load(open('indicators.yml'))

all_variables = [vdef.get('cmip5', vdef).get('name', vdef.get('name')) for vdef in indicators]
print(all_variables)

models = cmip5['models']

listing = []

# for experiment in cmip5['experiment']:
for experiment in ['rcp_8_5']:
    for model in models:
        for variable in all_variables:
            print(variable, model, experiment)
            request = CMIP5(variable, model, experiment, '200601-210012')
            try:
                if os.path.exists(request.downloaded_file):
                    print('file already present')
                else:
                    request.download()
                listing.append((variable, model, experiment))
            except:
                print('skip model', model)
                break

print('valid models for all asset', models)

json.dump({'listing':listing}, open('cmip5_listing.json','w'))
