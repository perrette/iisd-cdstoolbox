import os
import datetime
import netCDF4 as nc 
import pandas as pd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filenc')
    parser.add_argument('--units', default='days since 2000-01-01T06:00:00', help='%(default)s')
    o = parser.parse_args()

    ds = nc.Dataset(o.filenc)

    data = {}
    metadata = {}

    time = ds['time'][:]

    try:
        dates = nc.num2date(ds['time'][:], ds['time'].units, ds['time'].calendar)
        time = nc.date2num(dates, o.units, ds['time'].calendar) 
        units = o.units
    except Exception as error:
        print('!! error message:', str(error))
        print('!! failed to convert time units')
        units = 'unknown'

    data['time'] = time
    metadata['time'] = units

    for v in ds.variables:
        if v == 'time': 
            continue
        if ds[v].shape == ():
            metadata[v] = ds[v][:] 
            continue
        elif ds[v].shape != time.shape:
            print('!', v,'was ignored as it does not conform with time dimension')
            continue
        data[v] = ds[v][:]
        metadata[v] = getattr(ds[v], 'units', '')

    df = pd.DataFrame(data).set_index('time')
    base, ext = os.path.splitext(o.filenc)
    csvfile = base + '.csv'
    print('write to', csvfile)
    df.to_csv(csvfile)

    # add medata
    header = '\n'.join(
        '# {} : {}'.format(k, metadata[k])
        for k in metadata)

    txt = open(csvfile).read()
    open(csvfile,'w').write(header + '\n' + txt)


if __name__ == '__main__':
    main()