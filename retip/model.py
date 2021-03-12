import abc
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
import xgboost as xgb
import time

from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from retip import Dataset


class Trainer:
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def score(self, X_test=None, y_test=None):
        pass

    def predict(self, data):
        if isinstance(data, Dataset):
            X = data.get_data()
            y_pred = self.model.predict(X)
            y_series = pd.Series(y_pred, index=X.index)

            rt_col_idx = data.df.columns.get_loc(Dataset.RT_COLUMN)
            data.df(rt_col_idx + 1, 'RTP', y_series)
        elif isinstance(data, pd.DataFrame):
            return self.model.predict(data)
        else:
            print(f'Unsupported data format {type(data)}')


class XGBoostTrainer(Trainer):
    def __init__(self, dataset: Dataset, cv: int = 10, n_cpu: int = 4):
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
        X_train = training_data.drop('RT', axis=1)
        y_train = training_data.RT.values

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

    def score(self, X=None, y=None):
        if not X or not y:
            test_data = self.dataset.get_test_data()
            X = test_data.drop('RT', axis=1)
            y = test_data.RT.values
        

        y_pred = self.predict(X)
        rt_error = y - y_pred

        return {
            'root_mean_squared_error': metrics.mean_squared_error(y, y_pred, squared=True),
            'mean_absolute_error': metrics.mean_absolute_error(y, y_pred),
            'explained_variance_score': metrics.explained_variance_score(y, y_pred),
            'r2_score': metrics.r2_score(y, y_pred),
            'pearson_correlation': st.pearsonr(y, y_pred),
            'mean_squared_error': metrics.mean_squared_error(y, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y, y_pred),
            '95_percent_confidence_interval': st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error))
        }


class AutoGluonTrainer(Trainer):
    def __init__(self, dataset: Dataset, training_duration: int = 60, n_cpu: int = 4):
        self.dataset = dataset
        self.training_duration = training_duration
        self.n_cpu = n_cpu

        self.model = TabularPredictor(label=Dataset.RT_COLUMN)

    def train(self):
        t = time.time()

        self.model.fit(
            train_data=self.dataset.get_training_data(),
            time_limit=60 * self.minutes,
            preset='good_quality_faster_inference_only_refit',
            num_cpus=self.n_cpu
        )

        elapsed_time = str(datetime.timedelta(seconds=time.time() - t))

        fit_summary = model.fit_summary(verbosity=0)
        best_score = fit_summary['leaderboard'].loc[0, 'score']

        print(f'Training completed in {elapsed_time} with best RMSE {best_score:.3f}')

    def score(self, X=None, y=None):
        if not X or not y:
            test_data = self.dataset.get_test_data()
            X = test_data.drop('RT', axis=1)
            y = test_data.RT.values

        y_pred = self.predict(X)
        rt_error = y - y_pred

        return {
            'root_mean_squared_error': metrics.mean_squared_error(y, y_pred, squared=True),
            'mean_absolute_error': metrics.mean_absolute_error(y, y_pred),
            'explained_variance_score': metrics.explained_variance_score(y, y_pred),
            'r2_score': metrics.r2_score(y, y_pred),
            'pearson_correlation': st.pearsonr(y, y_pred),
            'mean_squared_error': metrics.mean_squared_error(y, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y, y_pred),
            '95_percent_confidence_interval': st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error))
        }
