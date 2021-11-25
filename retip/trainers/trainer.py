import abc
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn import metrics

import retip
from retip import Dataset


class Trainer:
    @abc.abstractmethod
    def save_model(self, filename: str):
        pass

    @abc.abstractmethod
    def load_model(self, filename: str):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def filter_columns(self, df):
        if hasattr(self, 'model_columns'):
            return df[[c for c in df.columns if c in self.model_columns]]
        else:
            return df

    def predict(self, data):
        if isinstance(data, Dataset):
            X = data.get_data()
            return self.model.predict(self.filter_columns(X))
        elif isinstance(data, pd.DataFrame):
            return self.model.predict(self.filter_columns(data))
        else:
            raise Exception(f'Unsupported data format {type(data)}')

    def score(self, data=None, plot: bool = None, plot_filename: str = None):
        if data is None:
            if self.dataset is not None:
                data = self.dataset.get_test_data()
            else:
                raise Exception('Trainer has no associated dataset and so it must be provided to the score method')
        elif isinstance(data, Dataset):
            data = data.get_data()
        elif isinstance(data, pd.DataFrame):
            pass
        else:
            raise Exception(f'Unsupported data format {type(data)}')

        y = data[Dataset.RT_COLUMN].values
        y_pred = self.predict(data)
        rt_error = y - y_pred

        if plot:
            retip.plot_rt_scatter(y, y_pred, output_filename=plot_filename)

        return {
            'root_mean_squared_error': metrics.mean_squared_error(y, y_pred, squared=False),
            'mean_absolute_error': metrics.mean_absolute_error(y, y_pred),
            'explained_variance_score': metrics.explained_variance_score(y, y_pred),
            'r2_score': metrics.r2_score(y, y_pred),
            'pearson_correlation': st.pearsonr(y, y_pred)[0],
            'mean_squared_error': metrics.mean_squared_error(y, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y, y_pred),
            '95_percent_confidence_interval': st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error))
        }

    def annotate(self, data):
        if isinstance(data, Dataset):
            X = data.get_data()

            if 'RTP' in X.columns:
                raise Exception('RTP column already exists!')
                return

            y_pred = self.model.predict(self.filter_columns(X))
            y_series = pd.Series(y_pred, index=X.index)

            if Dataset.RT_COLUMN in data.df.columns:
                idx = data.df.columns.get_loc(Dataset.RT_COLUMN)
            else:
                idx = data.df.columns.get_loc('SMILES')

            data.df.insert(idx + 1, 'RTP', y_series)
        elif isinstance(data, pd.DataFrame):
            if 'RTP' in data.columns:
                raise Exception('RTP column already exists!')
                return

            y_pred = self.model.predict(self.filter_columns(data))

            if Dataset.RT_COLUMN in data.columns:
                idx = data.df.columns.get_loc(Dataset.RT_COLUMN)
            elif 'SMILES' in data.columns:
                idx = data.df.columns.get_loc('SMILES')
            else:
                idx = 0

            data.insert(idx + 1, 'RTP', y_pred)
        else:
            raise Exception(f'Unsupported data format {type(data)}')
