import datetime
import pandas as pd
import shutil
import time

from autogluon.tabular import TabularPredictor

from retip import Dataset, Trainer


class AutoGluonTrainer(Trainer):
    def __init__(self, dataset: Dataset, training_duration: int = 60,
                 preset: str = 'good_quality_faster_inference_only_refit'):

        self.dataset = dataset
        self.training_duration = training_duration
        self.preset = preset


    def save_model(self, filename: str):
        if hasattr(self, 'model'):
            model_dir = self.model._learner.path
            shutil.move(model_dir, filename)

            print(f'Moved AutoGluon model to {filename}')
        else:
            raise Exception('Model has not been trained!')

    def load_model(self, filename: str):
        self.model = TabularPredictor.load(filename)


    def train(self):
        t = time.time()

        training_data = self.dataset.get_training_data()
        self.model_columns = list(training_data.drop(Dataset.RT_COLUMN, axis=1).columns)

        self.model = TabularPredictor(label=Dataset.RT_COLUMN)
        self.model.fit(
            train_data=training_data,
            time_limit=60 * self.training_duration,
            presets=self.preset
        )

        elapsed_time = str(datetime.timedelta(seconds=time.time() - t))

        fit_summary = pd.DataFrame(self.model.fit_summary(verbosity=0))
        best_score = -fit_summary.model_performance.max()

        print(f'Training completed in {elapsed_time} with best RMSE {best_score:.3f}')
