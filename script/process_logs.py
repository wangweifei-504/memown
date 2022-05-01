import os
import re
import warnings

import click
import pandas as pd

ATTRIBUTES = ['PR', 'REC', 'F1', 'ROC_ORI', 'mAP_ORI', 'ROC', 'mAP']


@click.command()
@click.option('--input-path', type=str)
@click.option('--dest-path', type=str)
@click.option('--model', type=click.Choice(['donut', 'bagel', 'usad']), default='donut')
# @click.option('--series', type=click.IntRange(0, 28), default=0)
@click.option('--label', type=click.Choice(['0', '01', '05']), default='0')
# @click.option('--seed', type=click.IntRange(2020, 2029), default=2020)
def process(input_path, dest_path, model, label):
    # files = os.listdir(path)
    # files = sorted(files)
    # print(files)
    if not os.path.exists(dest_path):
        warnings.warn(f'The save path {dest_path} dost not exist, created.')
        os.makedirs(dest_path)

    for series in range(0, 29):
        data = {attr: [] for attr in ATTRIBUTES}
        for seed in range(2020, 2030):
            if model == 'usad':
                file_name = f'{model}_s{series}_seed{seed}.txt'
            else:
                file_name = f'{model}_s{series}_label{label}_seed{seed}.txt'
            with open(os.path.join(input_path, file_name)) as f:
                content = f.read()
            # print(content)
            for attr in ATTRIBUTES:
                pattern = "'" + attr + "':" + r"\s?(\d+\.\d*)"
                # print(pattern)
                result = re.search(pattern, content)
                assert result is not None
                value = float(result.groups()[0])
                # print(value)
                data[attr].append(value)
        df = pd.DataFrame(data, index=range(2020, 2030))
        if model == 'usad':
            print(os.path.join(dest_path, f'{model}_series{series}.csv'))
            df.to_csv(os.path.join(dest_path, f'{model}_series{series}.csv'), index=True)
        else:
            df.to_csv(os.path.join(dest_path, f'{model}_series{series}_label{label}.csv'), index=True)
        print(df)


if __name__ == '__main__':
    process()
