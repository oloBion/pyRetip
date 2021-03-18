import datetime
import joblib
import xgboost as xgb
import time

from sklearn.model_selection import GridSearchCV

from retip import Dataset, Trainer



class XGBoostTrainer(Trainer):
    def __init__(self, dataset: Dataset = None, cv: int = 10, n_cpu: int = None):
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


    def save_model(self, filename: str):
        if hasattr(self, 'model'):
            export = {
                'model_name': 'XGBoost',
                'model_columns': self.model_columns,
                'model': self.model.best_estimator_
            }

            joblib.dump(export, filename)
            print(f'Exported model to {filename}')
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        export = joblib.load(filename)

        if isinstance(export, dict) and export.get('model_name') == 'XGBoost':
            self.model_columns = export['model_columns']
            self.model = export['model']
        else:
            raise Exception(f'{filename} is an invalid XGBoost model export')


    def train(self):
        if self.dataset is not None:
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
        else:
            raise Exception('Trainer has no associated dataset so it can only be used to predict new retention times')