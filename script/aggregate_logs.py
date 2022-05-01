import os
import warnings

import click
import numpy as np
import pandas as pd

ATTRIBUTES = ['PR', 'REC', 'F1', 'ROC_ORI', 'mAP_ORI', 'ROC', 'mAP']


@click.command()
@click.option('--input-path', type=str)
@click.option('--dest-path', type=str)
@click.option('--model', type=click.Choice(['donut', 'bagel', 'usad']), default='donut')
@click.option('--label', type=click.Choice(['0', '01', '05']), default='0')
def process(input_path, dest_path, model, label):
    if not os.path.exists(dest_path):
        warnings.warn(f'The save path {dest_path} dost not exist, created.')
        os.makedirs(dest_path)

    data = {}
    for series in range(0, 29):
        if model == 'usad':
            df = pd.read_csv(os.path.join(input_path, f'{model}_series{series}.csv'), index_col=0)
        else:
            df = pd.read_csv(os.path.join(input_path, f'{model}_series{series}_label{label}.csv'), index_col=0)
        print(df)
        for attr in df.columns:
            values = df.loc[:, attr].values
            if attr + '_mean' not in data.keys():
                data[attr + '_mean'] = []
            if attr + '_std' not in data.keys():
                data[attr + '_std'] = []
            data[attr + '_mean'].append(np.mean(values))
            data[attr + '_std'].append(np.std(values))
    df = pd.DataFrame(data)
    if model == 'usad':
        df.to_csv(os.path.join(dest_path, f'{model}_agg.csv'), float_format='%.4f')
    else:
        df.to_csv(os.path.join(dest_path, f'{model}_label{label}_agg.csv'), float_format='%.4f')
    print(df)


if __name__ == '__main__':
    process()
