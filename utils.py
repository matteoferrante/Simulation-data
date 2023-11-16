import warnings

import pandas as pd

import wandb


def get_runs(path):
    """Returns a pandas DataFrame which contains the runs' hyperparameters and
    metrics of the sweep identified by the given path.
    """

    api = wandb.Api()
    sweep = api.sweep(path)

    runs = []
    for run in sweep.runs:
        row = {}
        row.update(run.config), row.update(run.summary)
        row['name'] = run.name
        runs += [row]

    df = pd.DataFrame(runs)
    
    return df