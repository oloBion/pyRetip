import datetime
import importlib
import time

import h2o
from h2o.automl import H2OAutoML

from .. import Dataset
from . import Trainer


class H2OautoMLTrainer(Trainer):
    def __init__(self, dataset: Dataset = None, training_duration: int = None, max_models: int = 20, nfolds: int = -1):
        """
        """

        super().__init__(dataset)

        if not importlib.util.find_spec('h2o'):
            raise ImportError('H2O is not properly installed!')

        self.max_models = max_models
        self.nfolds = nfolds

        if training_duration is not None:
            self.training_duration = 60 * training_duration
        else:
            self.training_duration = 60

    def get_model(self, model_num: int = 0):
        lb_length = len(self.leaderboard)
        if 0 <= model_num < lb_length:
            mdl = h2o.get_model(self.leaderboard[model_num, "model_id"])
        else:
            raise Exception(f'Model has {lb_length} options. Select a model number between 0 and {lb_length-1}.')
        return mdl

    def save_model(self, model_num: int = 0, filename: str = None):
        if hasattr(self, 'predictor'):
            mdl = self.get_model(model_num)
            h2o.save_model(mdl, filename=filename, force=True)
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        h2o.init()
        self.predictor = h2o.load_model(filename)
        print(f'Loaded {filename}')

    def do_train(self):
        if self.dataset is not None:
            t = time.time()

            training_data = self.dataset.get_training_data()
            self.model_columns = list(filter(lambda x: x!=self.dataset.target_column,
                                             training_data.columns))

            h2o.init() # here?
            self.predictor = H2OAutoML(max_runtime_secs = self.training_duration,
                                       max_models=self.max_models,
                                       nfolds=self.nfolds)
            self.predictor.train(
                x = self.model_columns,
                y = self.dataset.target_column,
                training_frame = h2o.H2OFrame(training_data)
            )

            elapsed_time = str(datetime.timedelta(seconds=time.time() - t))

            self.leaderboard = self.predictor.leaderboard
            self.leader = self.predictor.leader

            best_score = self.leader.model_performance().rmse()

            print(f'Training completed in {elapsed_time} with best RMSE {best_score:.3f}')
        else:
            raise Exception('Trainer has no associated dataset so it can only be used to predict new retention times')

    def _predict(self, data):
        data = h2o.H2OFrame(data)
        with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
            predicted = self.predictor.predict(data).as_data_frame()["predict"].values
        return predicted