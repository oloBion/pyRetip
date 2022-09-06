import pandas as pd

from retip import Dataset, XGBoostTrainer


def test_xgboost():
    df = pd.read_csv('tests/data/data_retip.csv')
    data = Dataset().load_retip_dataset(df.head(10).copy())
    data.split_dataset(test_split=0.2, seed=123)

    data.calculate_descriptors()
    data.preprocess_features('metabolomics')
    assert data.get_training_data().shape[1] == 818

    trainer = XGBoostTrainer(data, cv=2)
    trainer.parameter_space = {}
    trainer.train()

    scores = trainer.score()
    print(scores)
