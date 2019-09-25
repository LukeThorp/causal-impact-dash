from causal_impact.causal_impact import CausalImpact

from datetime import date
from datetime import timedelta

import numpy as np
import pandas as pd

def run_ci_engine(data, date_inter, n_seasons=7):
    ci = CausalImpact(data, date_inter, n_seasons=n_seasons)
    result = ci.run(return_df=True)
    return ci, result

def mock_data(start_date, n_dates, inter_index, n_regressors):
    regressors = [
        np.random.rand() * 100 + 2 * np.random.rand() * np.random.randn(n_dates).cumsum()
        for _ in range(n_regressors)
    ]
    # Output from regressors + noise
    y = sum(2 * np.random.rand() * r for r in regressors) + np.random.randn(n_dates)
    # Adding artificial impact
    t = np.arange(n_dates - inter_index) / (n_dates - inter_index)
    i = np.random.rand() * 10000 * t ** 2 * np.exp(-n_dates / 10 * t)
    y[inter_index:] += i
    # Concatenating into a dataframe
    df = pd.concat(
        [pd.Series(r, name='x{}'.format(i)) for i, r in enumerate(regressors)] + [pd.Series(y, name='y')], axis=1
    )
    df.index = [start_date + timedelta(d) for d in range(n_dates)]
    return df

df = mock_data(date(2018, 1, 1), 30, 20, 2)

ci, result = run_ci_engine(df, date(2018, 1, 20), 5)

ci.plot()