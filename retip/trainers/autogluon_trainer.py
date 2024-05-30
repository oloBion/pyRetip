import datetime
import importlib
import pandas as pd
import shutil
import time

try:
    from autogluon.tabular import TabularPredictor
except:
    pass

from .. import Dataset
from . import Trainer


class AutoGluonTrainer(Trainer):
    def __init__(self, dataset: Dataset = None, training_duration: int = None, preset: str = 'high_quality'):
        """
        """

        super().__init__(dataset)

        if not importlib.util.find_spec('autogluon'):
            raise ImportError('AutoGluon is not properly installed!')

        self.preset = preset

        if training_duration is not None:
            self.training_duration = 60 * training_duration
        else:
            self.training_duration = None


    def save_model(self, filename: str):
        if hasattr(self, 'predictor'):
            model_dir = self.predictor._learner.path
            shutil.move(model_dir, filename)

            print(f'Moved AutoGluon model to {filename}')
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        self.predictor = TabularPredictor.load(filename)
        print(f'Loaded {filename}')


    def do_train(self):
        if self.dataset is not None:
            t = time.time()

            training_data = self.dataset.get_training_data()
            self.model_columns = list(filter(lambda x: x!=self.dataset.target_column,
                                             training_data.columns))

            self.predictor = TabularPredictor(label=self.dataset.target_column)
            self.predictor.fit(
                train_data=training_data,
                time_limit=self.training_duration,
                presets=self.preset
            )

            elapsed_time = str(datetime.timedelta(seconds=time.time() - t))

            fit_summary = pd.DataFrame(self.predictor.fit_summary(verbosity=0))
            best_score = -fit_summary.model_performance.max()

            print(f'Training completed in {elapsed_time} with best RMSE {best_score:.3f}')
        else:
            raise Exception('Trainer has no associated dataset so it can only be used to predict new retention times')
