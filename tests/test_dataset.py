import pandas as pd
import pytest

from retip import Dataset


def test_retip_dataset_csv_load():
    data = Dataset().load_retip_dataset('tests/data/data_retip.csv')
    assert len(data.get_training_data()) == 494

def test_retip_dataset_excel_load():
    data = Dataset().load_retip_dataset('tests/data/data_retip.xlsx')
    assert len(data.get_training_data()) == 494

def test_retip_dataset_df_load():
    df = pd.read_csv('tests/data/data_retip.csv')
    data = Dataset().load_retip_dataset(df)
    assert len(data.get_training_data()) == 494


def test_retip_dataset_csv_split():
    data = Dataset().load_retip_dataset('tests/data/data_retip.csv')
    data.split_dataset(test_split=0.2, seed=123)

    assert len(data.get_training_data()) == 395
    assert len(data.get_testing_data()) == 99

    with pytest.raises(Exception):
        data.split_dataset(test_split=0.2, seed=123)

    with pytest.raises(Exception):
        data.get_validation_data()

def test_retip_dataset_csv_split_validation():
    data = Dataset().load_retip_dataset('tests/data/data_retip.csv')
    data.split_dataset(test_split=0.2, validation_split=0.1, seed=123)

    assert len(data.get_training_data()) == 345
    assert len(data.get_testing_data()) == 99
    assert len(data.get_validation_data()) == 50


def test_retip_dataset_multi_csv_load():
    data = Dataset().load_retip_dataset('tests/data/data_retip_training.csv', 'tests/data/data_retip_testing.csv', 'tests/data/data_retip_validation.csv')

    assert len(data.get_training_data()) == 345
    assert len(data.get_testing_data()) == 99
    assert len(data.get_validation_data()) == 50


def test_gcn_dataset_load():
    data = Dataset().load_gcn_dataset('tests/data/data_gcn.csv')
    assert len(data.get_training_data()) == 345
    assert len(data.get_testing_data()) == 99
    assert len(data.get_validation_data()) == 50


def test_descriptor_calculation_min():
    df = pd.read_csv('tests/data/data_retip.csv')
    data = Dataset().load_retip_dataset(df.head(5).copy())
    assert len(data.get_training_data()) == 5

    data.calculate_descriptors()
    assert 'ABC' in data.get_training_data().columns

    data.preprocess_features('metabolomics')
    assert data.get_training_data().shape[1] == 818
    assert data.get_training_data(include_metadata=True).shape[1] == 821

    data.preprocess_features(['ABC'])
    assert data.get_training_data().shape[1] == 2
    assert data.get_training_data(include_metadata=True).shape[1] == 5

def test_descriptor_calculation_full_dataset():
    df = pd.read_csv('tests/data/data_retip.csv')
    data = Dataset().load_retip_dataset(df.head(10).copy())
    data.split_dataset(test_split=0.2, validation_split=0.2, seed=123)

    assert len(data.get_training_data()) == 6
    assert len(data.get_testing_data()) == 2
    assert len(data.get_validation_data()) == 2

    data.calculate_descriptors()
    assert 'ABC' in data.get_training_data().columns

    data.preprocess_features('metabolomics')
    assert data.get_training_data().shape[1] == 818

    data.preprocess_features(['ABC'])
    assert data.get_training_data().shape[1] == 2
