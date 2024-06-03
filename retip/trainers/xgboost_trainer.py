import datetime
import joblib
import xgboost as xgb
import pandas as pd
import time

from sklearn.model_selection import GridSearchCV

from .. import Dataset
from . import Trainer


class XGBoostTrainer(Trainer):
    def __init__(self, dataset: Dataset, cv: int = 10, n_cpu: int = None, n_jobs: int = -1):
        """
        """

        super().__init__(dataset)

        self.cv = cv
        self.n_cpu = n_cpu
        self.n_jobs = n_jobs

        self.parameter_space = {
            'n_estimators': [300, 400, 500, 600, 700, 800, 1000],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.02],
            'gamma': [1],
            'colsample_bytree': [0.5],
            'subsample': [0.5],
            'min_child_weight': [10]
        }


    def save_model(self, filename: str):
        if hasattr(self, 'predictor'):
            export = {
                'model_name': 'XGBoost',
                'model_columns': self.model_columns,
                'feature_importance': self.feature_importance,
                'predictor': self.predictor.best_estimator_
            }

            joblib.dump(export, filename)
            print(f'Exported model to {filename}')
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        export = joblib.load(filename)

        if isinstance(export, dict) and export.get('model_name') == 'XGBoost':
            self.model_columns = export['model_columns']
            self.feature_importance = export['feature_importance']
            self.predictor = export['predictor']
            print(f'Loaded {filename}')
        else:
            raise Exception(f'{filename} is an invalid XGBoost model export')


    def do_train(self, verbosity: int = 1):
        if self.dataset is not None:
            training_data = self.dataset.get_training_data()
            X_train = training_data.drop(self.dataset.target_column, axis=1)
            y_train = training_data[self.dataset.target_column].values

            self.model_columns = X_train.columns

            t = time.time()

            self.predictor = GridSearchCV(
                xgb.XGBRegressor(n_jobs=self.n_cpu),
                self.parameter_space,
                cv=self.cv,
                verbose=verbosity,
                n_jobs=self.n_jobs
            ).fit(X_train, y_train)

            self.feature_importance = self.predictor.best_estimator_.feature_importances_

            elapsed_time = str(datetime.timedelta(seconds=time.time() - t))
            print(f'Training completed in {elapsed_time} with best RMSE {self.predictor.best_score_:.3f}')
        else:
            raise Exception('Trainer has no associated dataset so it can only be used to predict new retention times')

    def get_feature_importance(self):
        if hasattr(self, 'predictor'):
            df = pd.DataFrame({"feature": self.model_columns,
                               "importance": self.feature_importance})
            df.sort_values(by="importance", ascending=False, inplace=True)
            df.reset_index(inplace=True, drop=True)
            return df
        else:
            raise Exception('Model has not been trained!')
