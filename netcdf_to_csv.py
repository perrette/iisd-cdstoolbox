import os
import datetime
import itertools
import numpy as np
import netCDF4 as nc 
import pandas as pd

def convert_time(ds, units='days since 2000-01-01'):
    try:
        dates = nc.num2date(ds['time'][:], ds['time'].units, ds['time'].calendar)
        time = nc.date2num(dates, units, ds['time'].calendar) 
    except Exception as error:
        print('!! error message:', str(error))
        print('!! failed to convert time units')
        time = ds['time'][:]
        units = 'unknown'
    return time, units


def process_file(filenc, units):

    ds = nc.Dataset(filenc)

    data = {}
    metadata = {}

    time, units = convert_time(ds, units)

    data['time'] = time
    metadata['time'] = units

    for v in ds.variables:
        if v == 'time': 
            continue

        # scalar: just add as metadata
        if ds[v].shape == ():
            metadata[v] = ds[v][:] 
            continue

        # check !
        elif 'time' not in ds[v].dimensions:
            if ds[v].ndim == 1:
                metadata[v] = ','.join(str(x) for x in ds[v][:])
            else:
                print('!', v,'was added as metadata as it does not conform with time dimension')
            continue

        # multi-dimensional array (model ensemble)
        elif ds[v].shape != time.shape:
            dims = ds[v].dimensions
            assert dims[-1] == 'time', 'time must be the last dim'
            a = ds[v][:]
            i_axes = [np.arange(i) for i in a.shape[:-1]]
            axes = [ds[a][:] if a in ds.variables else i_axes[i] for i, a in enumerate(dims[:-1])]
            for indices in itertools.product(*i_axes):
                key = '-'.join([str(i_axes[i][idx]) for i, idx in enumerate(indices)])
                data[key] = a[indices]
            metadata[v] = getattr(ds[v], 'units', '')

            metadata['dimensions'] = '''
{}'''.format('\n'.join(['# - {}: {}'.format(d, len(ds.dimensions[d])) for d in ds[v].dimensions]))

        # normal case, main variable
        else:
            data[v] = ds[v][:]
            metadata[v] = getattr(ds[v], 'units', '')


    df = pd.DataFrame(data).set_index('time')
    base, ext = os.path.splitext(filenc)
    csvfile = base + '.csv'
    print('write to', csvfile)
    df.to_csv(csvfile)

    # add medata
    header = '\n'.join(
        '# {} : {}'.format(k, metadata[k])
        for k in metadata)

    txt = open(csvfile).read()
    open(csvfile,'w').write(header + '\n' + txt)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filenc', nargs='+')
    parser.add_argument('--units', default='days since 2000-01-01', help='%(default)s')
    o = parser.parse_args()

    for f in o.filenc:
        print('extract', f)
        process_file(f, o.units)

if __name__ == '__main__':
    main()
