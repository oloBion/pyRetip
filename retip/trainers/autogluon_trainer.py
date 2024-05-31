import datetime
import importlib
import pandas as pd
import shutil
import time
from typing import Union

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

    def feature_importance(self, data: Union[Dataset, pd.DataFrame] = None):
        if hasattr(self, 'predictor'):
            if data is None:
                if self.dataset is not None:
                    data = self.dataset.get_training_data()
                else:
                    raise Exception('Trainer has no associated dataset and so it must be provided to get feature importance')
            elif isinstance(data, Dataset):
                raise Exception('Please specify one dataset from the training, testing, or validation subsets of the Dataset object.')
            elif isinstance(data, pd.DataFrame):
                pass
            else:
                raise Exception(f'Unsupported data format {type(data)}')

            df = self.predictor.feature_importance(data)
            df.sort_values(by="importance", ascending=False, inplace=True)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "feature"}, inplace=True)
            df = df.loc[:, ("feature", "importance")]
            return df
        else:
            raise Exception('Model has not been trained!')
