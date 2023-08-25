import abc
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.metrics as metrics

from typing import Union

from .. import Dataset



class Trainer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.predictor = None
        self.model_columns = None


    @abc.abstractmethod
    def save_model(self, filename: str):
        pass

    @abc.abstractmethod
    def load_model(self, filename: str):
        pass

    @abc.abstractmethod
    def do_train(self):
        pass

    def train(self):
        for k, df in self.dataset.datasets.items():
            # ensure that the target column is present
            if self.target_column not in df.columns:
                raise Exception(f'Target column "{self.target_column}" was not found in the {k} dataset')

        self.do_train()

    def filter_columns(self, df):
        if self.model_columns is not None:
            return df[[c for c in df.columns if c in self.model_columns]]
        else:
            return df

    def predict(self, data):
        """
        """

        if isinstance(data, Dataset):
            X = data.get_training_data()
            return self.predictor.predict(self.filter_columns(X))
        elif isinstance(data, pd.DataFrame):
            return self.predictor.predict(self.filter_columns(data))
        else:
            raise Exception(f'Unsupported data format {type(data)}')

    def score(self, data: Union[Dataset, pd.DataFrame] = None, target_column: str = None, plot: bool = None, plot_filename: str = None):
        """
        """

        if data is None:
            if self.dataset is not None:
                data = self.dataset.get_testing_data()
                target_column = self.dataset.target_column
            else:
                raise Exception('trainer has no associated dataset and so it must be provided to the score method')
        elif isinstance(data, Dataset):
            data = data.get_data()
            target_column = data.target_column
        elif isinstance(data, pd.DataFrame):
            if target_column is None:
                raise Exception('target column name must be provided when scoring a data frame')
        else:
            raise Exception(f'unsupported data format {type(data)}')

        y = data[target_column].values
        y_pred = self.predict(data)
        rt_error = y - y_pred

        if plot:
            from .. import visualization
            visualization.plot_rt_scatter(y, y_pred, output_filename=plot_filename)

        epsilon = np.finfo(np.float64).eps

        return {
            'root_mean_squared_error': metrics.mean_squared_error(y, y_pred, squared=False),
            'mean_squared_error': metrics.mean_squared_error(y, y_pred),
            'mean_absolute_error': metrics.mean_absolute_error(y, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y, y_pred),
            'explained_variance_score': metrics.explained_variance_score(y, y_pred),

            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error(y, y_pred),
            'absolute_median_relative_error': np.median((abs(y - y_pred) / np.maximum(y, epsilon))),

            'r2_score': metrics.r2_score(y, y_pred),
            'pearson_correlation': st.pearsonr(y, y_pred)[0],

            '90_percent_confidence_interval': st.norm.ppf(0.9, loc=np.mean(rt_error), scale=np.std(rt_error)),
            '95_percent_confidence_interval': st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error))
        }

    def annotate(self, df: pd.DataFrame, prediction_column: str):
        """
        """

        if isinstance(df, pd.DataFrame):
            if prediction_column in df.columns:
                raise Exception(f'{prediction_column} column already exists!')

            y_pred = self.predictor.predict(self.filter_columns(df))

            if 'SMILES' in df.columns:
                idx = df.columns.get_loc('SMILES')
            else:
                idx = 0

            df = df.copy()
            df.insert(idx + 1, prediction_column, y_pred)
            return df
        else:
            raise Exception(f'unsupported data format {type(df)}')
