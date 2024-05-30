import datetime
import joblib
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .. import Dataset
from . import Trainer


class RandomForestTrainer(Trainer):
    def __init__(self, dataset: Dataset, n_estimators: int = 100, random_state: int = 0):
        """
        """

        super().__init__(dataset)

        self.n_estimators = n_estimators
        self.random_state = random_state

    def save_model(self, filename: str):
        if hasattr(self, 'predictor'):
            export = {
                'model_name': 'RandomForest',
                'model_columns': self.model_columns,
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
            self.predictor = export['model']
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

            self.predictor = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    verbose=verbosity,
                    oob_score=True,
                ).fit(X_train, y_train)
            elapsed_time = str(datetime.timedelta(seconds=time.time() - t))
            # ? rmse = mean_squared_error(y_train, self.predictor.oob_prediction_, squared=False)
            print(f'Training completed in {elapsed_time} with best R\U000000B2 {self.predictor.oob_score_:.3f}')
        else:
            raise Exception('Trainer has no associated dataset so it can only be used to predict new retention times')
        
