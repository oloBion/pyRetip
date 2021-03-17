import abc
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
import xgboost as xgb
import time

from autogluon.tabular import TabularPredictor

from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from retip import Dataset


class Trainer:
    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def predict(self, data):
        if isinstance(data, Dataset):
            X = data.get_data()
            return self.model.predict(X[self.model_columns])
        elif isinstance(data, pd.DataFrame):
            return self.model.predict(data[self.model_columns])
        else:
            print(f'Unsupported data format {type(data)}')

    def score(self, data=None):
        if data is None:
            data = self.dataset.get_test_data()
        elif isinstance(data, Dataset):
            data = data.get_data()
        elif isinstance(data, pd.DataFrame):
            pass
        else:
            print(f'Unsupported data format {type(data)}')
            return

        y = data[Dataset.RT_COLUMN].values
        y_pred = self.predict(data)
        rt_error = y - y_pred

        return {
            'root_mean_squared_error': metrics.mean_squared_error(y, y_pred, squared=True),
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
                print('RTP column already exists!')
                return

            y_pred = self.model.predict(X[self.model_columns])
            y_series = pd.Series(y_pred, index=X.index)

            if Dataset.RT_COLUMN in data.df.columns:
                idx = data.df.columns.get_loc(Dataset.RT_COLUMN)
            else:
                idx = data.df.columns.get_loc('SMILES')

            data.df.insert(idx + 1, 'RTP', y_series)
        elif isinstance(data, pd.DataFrame):
            if 'RTP' in data.columns:
                print('RTP column already exists!')
                return

            y_pred = self.model.predict(data[self.model_columns])

            if Dataset.RT_COLUMN in data.columns:
                idx = data.df.columns.get_loc(Dataset.RT_COLUMN)
            elif 'SMILES' in data.columns:
                idx = data.df.columns.get_loc('SMILES')
            else:
                idx = 0

            data.insert(idx + 1, 'RTP', y_pred)
        else:
            print(f'Unsupported data format {type(data)}')


class XGBoostTrainer(Trainer):
    def __init__(self, dataset: Dataset, cv: int = 10, n_cpu: int = None):
        self.dataset = dataset
        self.cv = cv
        self.n_cpu = n_cpu

        self.parameter_space = {
            'n_estimators': [300, 400, 500, 600, 700, 800, 1000],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.02],
            'gamma': [1],
            'colsample_bytree': [0.5],
            'subsample': [0.5],
            'min_child_weight': [10]
        }

    def train(self):
        training_data = self.dataset.get_training_data()
        X_train = training_data.drop(Dataset.RT_COLUMN, axis=1)
        y_train = training_data[Dataset.RT_COLUMN].values

        self.model_columns = X_train.columns

        t = time.time()

        self.model = GridSearchCV(
            xgb.XGBRegressor(n_jobs=self.n_cpu),
            self.parameter_space,
            cv=self.cv,
            verbose=1,
            n_jobs=1
        ).fit(X_train, y_train)

        elapsed_time = str(datetime.timedelta(seconds=time.time() - t))
        print(f'Training completed in {elapsed_time} with best RMSE {self.model.best_score_:.3f}')


class AutoGluonTrainer(Trainer):
    def __init__(self, dataset: Dataset, training_duration: int = 60,
                 preset: str = 'good_quality_faster_inference_only_refit'):

        self.dataset = dataset
        self.training_duration = training_duration
        self.preset = preset

        self.model = TabularPredictor(label=Dataset.RT_COLUMN)

    def train(self):
        t = time.time()

        training_data = self.dataset.get_training_data()
        self.model_columns = list(training_data.drop(Dataset.RT_COLUMN, axis=1).columns)

        self.model.fit(
            train_data=training_data,
            time_limit=60 * self.training_duration,
            presets=self.preset
        )

        elapsed_time = str(datetime.timedelta(seconds=time.time() - t))

        fit_summary = model.fit_summary(verbosity=0)
        best_score = fit_summary['leaderboard'].loc[0, 'score']

        print(f'Training completed in {elapsed_time} with best RMSE {best_score:.3f}')
