import abc
import datetime
import numpy as np
import scipy.stats as st
import xgboost as xgb
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from retip import Dataset


class Trainer:
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def score(self, X_test=None, y_test=None):
        pass

    @abc.abstractmethod
    def predict(self):
        pass


class XGBoostTrainer(Trainer):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

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
            xgb.XGBRegressor(n_jobs=8),
            self.parameter_space,
            cv=10,
            verbose=1,
            n_jobs=1
        ).fit(X_train, y_train)

        print('Completed in', str(datetime.timedelta(seconds=time.time() - t)))
        print(self.model.best_score_)
        print(self.model.best_params_)
        print()

    
    def score(self, X=None, y=None):
        if not X or not y:
            test_data = self.dataset.get_test_data()
            X = test_data.drop('RT', axis=1)
            y = test_data.RT.values
        

        y_pred = self.model.predict(X)
        rt_error = y - y_pred

        print('RMSE:', mean_squared_error(y, y_pred, squared=False))
        print('R^2:', r2_score(y, y_pred))
        print('MAE', mean_absolute_error(y, y_pred))
        print('95% CI:', st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error)))
