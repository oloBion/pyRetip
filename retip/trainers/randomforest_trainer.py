import datetime
import joblib
import time
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from .. import Dataset
from . import Trainer


class RandomForestTrainer(Trainer):
    def __init__(self, dataset: Dataset, cv: int = 10, n_jobs: int = -1, random_state: int = 50):
        """
        """

        super().__init__(dataset)

        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.parameter_space = {
            'n_estimators': [int(x) for x in range(100, 1001, 100)],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [int(x) for x in range(10, 101, 10)] + [None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

    def save_model(self, filename: str):
        if hasattr(self, 'predictor'):
            export = {
                'model_name': 'RandomForest',
                'model_columns': self.model_columns,
                'feature_importance': self.feature_importance,
                'predictor': self.predictor
            }
            joblib.dump(export, filename)
            print(f'Exported model to {filename}')
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        export = joblib.load(filename)

        if isinstance(export, dict) and export.get('model_name') == 'RandomForest':
            self.model_columns = export['model_columns']
            self.feature_importance = export['feature_importance']
            self.predictor = export['predictor']
            print(f'Loaded {filename}')
        else:
            raise Exception(f'{filename} is an invalid Random Forest model export')

    def do_train(self, verbosity: int = 1):
        if self.dataset is not None:
            training_data = self.dataset.get_training_data()
            X_train = training_data.drop(self.dataset.target_column, axis=1)
            y_train = training_data[self.dataset.target_column].values

            self.model_columns = X_train.columns

            t = time.time()

            # Define the RandomizedSearchCV
            self.predictor = RandomizedSearchCV(
                estimator= RandomForestRegressor(),
                param_distributions=self.parameter_space,
                n_iter=10,
                scoring='r2',
                cv=self.cv,
                verbose=verbosity,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ).fit(X_train, y_train)

            self.feature_importance = self.predictor.best_estimator_.feature_importances_

            elapsed_time = str(datetime.timedelta(seconds=time.time() - t))
            print(f'Training completed in {elapsed_time} with best R\U000000B2 {self.predictor.best_score_}')
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
