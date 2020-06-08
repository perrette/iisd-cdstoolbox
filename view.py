import matplotlib.pyplot as plt

from pathlib import Path
import re, os
from common import load_csv, convert_time_units_series

def get_metadata(csv_file):
    path = Path(csv_file)

    name, _ = os.path.splitext(path.name)
    simulation_set = path.parents[0].name if len(path.parents) else 'unknown'
    asset = path.parents[1].name if len(path.parents) > 1 else 'unknown'
    location = path.parents[2].name if len(path.parents) > 2 else 'unknown'
    return {'name': name, 'simulation_set': simulation_set, 'asset': asset, 'location': location}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', nargs='+')
    parser.add_argument('--yearly-mean', action='store_true')
    parser.add_argument('--png', help='file name for saving png figure')
    o = parser.parse_args()

    fig2 = plt.figure()
    ax2 = plt.subplot(1, 1, 1)

    # compare string to find label
    records = [get_metadata(csv_file) for csv_file in o.csv_file]
    keep_label = {field: len(set(r[field] for r in records)) > 1 for field in records[0]}

    for csv_file in o.csv_file:
        ts = load_csv(csv_file)
        ts.index = convert_time_units_series(ts.index, years=True)

        record = get_metadata(csv_file)
        label = ', '.join(str(value) for key, value in record.items() if keep_label.get(key))

        # name = ts.columns[1]
        path = Path(csv_file)

        name = path.name
        cname = ts.name
        color = None
        zorder = None

        # units = re.match(r'', ts.columns[1])
        try:
            units, = re.match(r'.* \((.*)\)', cname).groups()
        except:
            logging.warning(f'failed to parse units: {cname}')
            units = ''

        l, = ax2.plot(ts.index, ts.values, alpha=0.5, label=label, linewidth=1 if o.yearly_mean else 2, color=color, zorder=zorder)

        # add yearly mean as well
        if o.yearly_mean:
            yearly_mean = ts.rolling(12).mean()
            l2, = ax2.plot(ts.index[::12], yearly_mean[::12], alpha=1, linewidth=2, color=l.get_color(), zorder=zorder)

    ax2.legend(fontsize='xx-small')
    ax2.set_ylabel(units)
    ax2.set_xlabel(ts.index.name)
    ax2.set_title(name)
    # ax2.set_xlim(xmin=start_year, xmax=2100)

    mi, ma = ax2.get_xlim()
    if mi < 0:
        ax2.set_xlim(xmin=0)  # start at start_year (i.e. ERA5 start)

    if o.png:
        figname = o.png
        fig2.savefig(figname, dpi=300)

    else:
        plt.show()


if __name__ == '__main__':
    main()