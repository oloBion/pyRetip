from curses import meta
import numpy as np
import json
import pandas as pd
import pkgutil
import pubchempy as pcp
import tqdm
import warnings

from pathlib import Path
from typing import List, Union

from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.model_selection import train_test_split


RT_COLUMN = 'RT'
IDENTIFIER_COLUMNS = ['PubChem CID', 'SMILES']


class Dataset:
    def __init__(self, target_column: str = RT_COLUMN):
        """
        """

        self.target_column = target_column
        self.datasets = {}

        # create mordred calculator
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.descriptor_names = [str(d) for d in self.calc.descriptors]


    def load_retip_dataset(self, training: Union[str, Path, pd.DataFrame], testing: Union[str, Path, pd.DataFrame] = None,
                           validation: Union[str, Path, pd.DataFrame] = None, training_sheet_name: Union[str, int] = 0,
                           testing_sheet_name: Union[str, int] = 0, validation_sheet_name: Union[str, int] = 0):
        """
        """

        self.datasets = {}
        self.datasets['training'] = self._load_dataframe(training, training_sheet_name)

        if testing is not None:
            self.datasets['testing'] = self._load_dataframe(testing, testing_sheet_name)

        if validation is not None:
            self.datasets['validation'] = self._load_dataframe(validation, validation_sheet_name)

        return self._validate_dataframe()

    def load_gcn_dataset(self, dataset: Union[str, Path, pd.DataFrame]):
        """
        """

        df = self._load_dataframe(dataset)
        df = df.rename({'compound_name': 'Name', 'retention_time': 'RT', 'smiles': 'SMILES'}, axis=1)
        split_index = df.pop('split_index')

        self.datasets = {}
        self.datasets['training'] = df[split_index.isin([1, 2])].reset_index(drop=True)
        self.datasets['testing'] = df[split_index == 3].reset_index(drop=True)

        if (split_index == 4).any():
            self.datasets['validation'] = df[split_index == 4].reset_index(drop=True)

        return self._validate_dataframe()

    def _load_dataframe(self, dataset: Union[str, Path, pd.DataFrame], sheet_name: Union[str, int] = 0):
        """
        """

        if isinstance(dataset, str):
            if dataset.lower().endswith('.csv'):
                return pd.read_csv(dataset)
            elif dataset.lower().endswith('.xls') or dataset.lower().endswith('.xlsx'):
                return pd.read_excel(dataset, sheet_name=sheet_name)
            else:
                extension = dataset.split('.')[-1]
                raise Exception(f'{extension} is not a supported data format')

        elif isinstance(dataset, pd.DataFrame):
            return dataset

        else:
            raise Exception(f'{type(dataset)} is not a supported data type')

    def _validate_dataframe(self):
        """
        """

        for k, df in self.datasets.items():
            # ensure that the target column is present
            if self.target_column not in df.columns:
                raise Exception(f'Target column "{self.target_column}" was not found in the {k} dataset')

            # ensure that at least one identifier column is present
            identifier_cols = [x for x in IDENTIFIER_COLUMNS if x in df.columns]

            if not any(x in df.columns for x in IDENTIFIER_COLUMNS):
                valid_columns = ', '.join(IDENTIFIER_COLUMNS)
                raise Exception(f'No identifier columns ({valid_columns}) were not found in the {k} dataset')

            # clean identifier columns
            for col in identifier_cols:
                df.loc[:, col] = df[col].str.strip().replace('', np.NaN)

            # check for missing identifiers
            missing_identifiers = df[identifier_cols].isnull().any(axis=1).sum()

            if missing_identifiers:
                print(f'Warning: the {k} dataset is missing {missing_identifiers} structural identifiers - these rows will be ignored')

        return self


    def save_retip_dataset(self, filename_prefix: str, include_descriptors: bool = True):
        """
        """

        for k, df in self.datasets.items():
            if not include_descriptors:
                df = df[df.columns.difference(self.descriptor_names)]

            df.to_csv(f'{filename_prefix}_{k}.csv', index=False)
            print(f'Saved {k} dataset to {filename_prefix}_{k}.csv')

    def split_dataset(self, test_split: float = 0.2, validation_split: float = 0, seed: int = None):
        """
        """

        if 'testing' in self.datasets:
            raise Exception('Dataset is already split into training and test subsets')
        elif test_split == 0:
            raise Exception('A test set split larger than 0 must be given')
        elif validation_split > 0 and 'validation' in self.datasets:
            raise Exception('A validation dataset has already been provided')
        elif test_split + validation_split > 0.4:
            raise Exception('training set must consist of at least 40% of the dataset')
        else:
            split_ratio = test_split + validation_split
            self.datasets['training'], self.datasets['testing'] = train_test_split(self.datasets['training'], test_size=split_ratio, random_state=seed)

            if validation_split > 0:
                split_ratio = validation_split / (test_split + validation_split)
                self.datasets['testing'], self.datasets['validation'] = train_test_split(self.datasets['testing'], test_size=split_ratio, random_state=seed)


    def _pull_pubchem_smiles(self, pubchem_cid: int):
        """
        """

        try:
            c = pcp.Compound(pubchem_cid)
            return c.canonical_smiles
        except pcp.NotFoundError:
            print(f'Error: no record found for PubChem CID {pubchem_cid}')

    def calculate_descriptors(self, n_proc: int = 1):
        """
        """

        for k, df in self.datasets.items():
            if any(c in self.descriptor_names for c in df.columns):
                print(f'Skipping descriptor calculation for {k} dataset, descriptors have already been calculated')
                continue

            print(f'Calculating descriptors for {k} dataset')
            descs = []

            for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
                smi = row.SMILES

                # pull SMILES PubChem if necessary
                if pd.isnull(row.SMILES):
                    if not pd.isnull(row['PubChem CID']):
                        smi = self._pull_pubchem_smiles(row['PubChem CID'])

                        if smi:
                            df.loc[i, 'SMILES'] = smi

                # calculate chemical descriptors if necessary
                if pd.isnull(smi):
                    descs.append({})
                else:
                    try:
                        mol = Chem.MolFromSmiles(smi)

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=DeprecationWarning)
                            warnings.filterwarnings('ignore', category=RuntimeWarning)

                            desc = self.calc(mol)
                            desc = desc.fill_missing()
                            desc = desc.asdict()
                            descs.append(desc)
                    except:
                        print(f'Error: descriptor calculation failed for {smi}')
                        descs.append({})
            
            descs = pd.DataFrame(descs)
            descs = descs.replace({False: 0, True: 1})

            self.datasets[k] = df.join(descs)


    def preprocess_features(self, descriptor_subset: Union[str, List[str]] = None, descriptor_counts_threshold: float = 0.999):
        """
        """

        dataset_columns = []

        # preprocess each dataset indiviually
        for k, df in self.datasets.items():
            descriptor_cols = [c for c in df.columns if c in self.descriptor_names]

            # remove rows with no non-null descriptors
            df = df.dropna(how='all', subset=descriptor_cols)

            # separate metdaata and features
            metadata = df.loc[:, [c for c in df.columns if c not in descriptor_cols]]
            features = df.loc[:, descriptor_cols]

            # filter descriptors by predefined subset if specified
            if descriptor_subset is None:
                pass
            elif isinstance(descriptor_subset, str):
                # use only descriptors calculable for a given fraction of the given chemical dataset
                if descriptor_subset in ['metabolomics', 'lipidomics']:
                    descriptor_counts = json.loads(pkgutil.get_data(__package__, f'data/{descriptor_subset}_descriptors_counts.json'))
                    max_count = max(descriptor_counts.values())

                    descriptor_subset = [c for c, v in descriptor_counts.items() if v >= descriptor_counts_threshold * max_count]
                    features = features.loc[:, [c for c in descriptor_cols if c in descriptor_subset]]
                else:
                    raise Exception(f'invalid descriptor subset name {descriptor_subset}')
            elif isinstance(descriptor_subset, list):
                # use custom feature inclusion list
                features = features.loc[:, [c for c in descriptor_cols if c in descriptor_subset]]
            else:
                raise Exception(f'invalid descriptor subset type {descriptor_subset}')
            
            # drop descriptors with any null values
            features = features.dropna(how='any', axis=1)

            # update dataset
            self.datasets[k] = metadata.join(features)
            dataset_columns.append(list(self.datasets[k].columns))


        # restrict features to those shared by all datasets
        columns = set.intersection(*map(set, dataset_columns))
        columns = sorted(columns, key=dataset_columns[0].index)

        for k in self.datasets:
            self.datasets[k] = self.datasets[k].loc[:, columns]
        
        print(f'Reduced feature set from {len(descriptor_cols)} to {len(columns) - len(metadata.columns)}')


    def head(self, n: int = 5):
        """
        """
        
        if not self.datasets:
            raise Exception('No datasets defined!')

        for k, df in self.datasets.items():
            print(k.title())
            print(df.head(n))
            print()

    def describe(self):
        """
        """
        
        if not self.datasets:
            raise Exception('No datasets defined!')

        for k, df in self.datasets.items():
            print(k.title(), df.shape)


    def _get_dataset(self, dataset_name: str, include_metadata: bool = False):
        if dataset_name not in self.datasets:
            raise Exception(f'No {dataset_name} dataset defined!')

        if include_metadata:
            return self.datasets[dataset_name]
        else:
            df = self.datasets[dataset_name]
            return df.loc[:, [self.target_column] + [c for c in df.columns if c in self.descriptor_names]]

    def has_training_data(self):
        return 'training' in self.datasets

    def get_training_data(self, include_metadata: bool = False):
        return self._get_dataset('training', include_metadata)

    def has_testing_data(self):
        return 'testing' in self.datasets

    def get_testing_data(self, include_metadata: bool = False):
        return self._get_dataset('testing', include_metadata)

    def has_validation_data(self):
        return 'validation' in self.datasets

    def get_validation_data(self, include_metadata: bool = False):
        return self._get_dataset('validation', include_metadata)
