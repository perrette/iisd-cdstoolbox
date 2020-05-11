"""Download variables related to one asset class
"""
import os
import cdsapi


class CMIP5:
    def __init__(self, variable, model, scenario, period, ensemble=None):
        self.variable = variable
        self.model = model
        self.scenario = scenario
        self.period = period
        self.ensemble = ensemble or 'r1i1p1'

        self.folder = os.path.join('download', 'cmip5')
        self.name = '{variable}-{model}-{scenario}-{period}-{ensemble}'.format(**vars(self)) 
        self.downloaded_file = os.path.join(self.folder, self.name+'.zip')


    def download(self):
        c = cdsapi.Client()

        os.makedirs(self.folder, exist_ok=True)

        print('download', self.downloaded_file, '...')

        res = c.retrieve(
            'projections-cmip5-monthly-single-levels',
            {
                'ensemble_member': self.ensemble,
                'format': 'zip',
                'experiment': self.scenario,
                'variable': self.variable,
                'model': self.model,
                'period': self.period,
            },
            self.downloaded_file)
        return res


def main():
    temp = CMIP5('2m_temperature', 'ipsl_cm5a_mr', 'rcp_8_5', '200601-210012')

    if not os.path.exists(temp.downloaded_file):
        temp.download()

if __name__ == '__main__':
    main()