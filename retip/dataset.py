import numpy as np
import pandas as pd
import tqdm

from mordred import Calculator, descriptors
from rdkit import Chem

from sklearn.model_selection import train_test_split


class Dataset:
    NAME_COLUMN = 'Name'
    RT_COLUMN = 'RT'
    IDENTIFIER_COLUMNS = ['PubChem CID', 'SMILES']

    def __init__(self, filename, test_size: float = 0.2, seed: int = None):
        # load and validate data set
        self.load_dataframe(filename)

        self.seed = seed
        self.test_size = test_size
        self.data = None

        # create mordred calculator
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.descriptor_names = [str(d) for d in self.calc.descriptors]


    def load_dataframe(self, filename):
        """
        """

        # load data frame
        if filename.lower().endswith('.csv'):
            self.df = pd.read_csv(filename)
        elif filename.lower().endswith('.xlsx'):
            self.df = pd.read_excel(filename)
        else:
            extension = filename.split('.')[-1]
            raise Exception(f'{extension} is not a supported data format')

        # ensure that the data frame contains required columns
        if self.NAME_COLUMN not in self.df.columns:
            raise Exception(f'{self.NAME_COLUMN} column was not found in the data frame')
        
        if self.RT_COLUMN not in self.df.columns:
            raise Exception(f'{self.RT_COLUMN} column was not found in the data frame')

        if not any(x in self.df.columns for x in self.IDENTIFIER_COLUMNS):
            valid_columns = ', '.join(self.IDENTIFIER_COLUMNS)
            raise Exception(f'No identifier columns were not found in the data frame: {valid_columns}')

    def load_structures(self):
        """
        """

    
    def calculate_descriptors(self):
        """
        """

        assert('SMILES' in self.df.columns)

        if all(d in self.df.columns for d in self.descriptor_names):
            print('Skipping molecular descriptor calculation, descriptors have already been calculated')

        descs = []

        for smi in tqdm.tqdm(set(self.df.SMILES)):
            try:
                mol = Chem.MolFromSmiles(smi)

                desc = self.calc(mol)
                desc = desc.fill_missing()

                desc = desc.asdict()
                desc['SMILES'] = smi

                descs.append(desc)
            except:
                print(f'Parsing SMILES {smi} failed')

        descs = pd.DataFrame(descs)
        self.df = pd.merge(self.df, descs, how='left', on='SMILES')


    def build_dataset(self):
        descriptor_names = [str(d) for d in self.calc.descriptors]

        if not all(d in self.df.columns for d in self.descriptor_names):
            self.calculate_descriptors()
        
        data = self.df[[self.RT_COLUMN] + self.descriptor_names]
        data = data.dropna(how='all', subset=self.descriptor_names)
        data = data.dropna(how='any', axis=1)

        self.training_data, self.test_data = train_test_split(data, test_size=self.test_size, random_state=self.seed)
    
    def save_dataset(self, filename):
        if filename.lower().endswith('.csv'):
            self.df.to_csv(filename, index=False)
        elif filename.lower().endswith('.xlsx'):
            self.df.to_excel(filename, index=False)
        else:
            extension = filename.split('.')[-1]
            raise Exception(f'{extension} is not a supported data format')

        print(f'Saved dataset to {filename}')


    def get_training_data(self):
        if not self.data:
            self.build_dataset()
        
        return self.training_data
    
    def get_test_data(self):
        if not self.data:
            self.build_dataset()
        
        return self.test_data
