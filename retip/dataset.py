import mordred
import pandas as pd

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
    

    def build_dataset(self):
        descriptor_names = [str(d) for d in self.calc.descriptors]

        if not all(d in self.df.columns for d in descriptor_names):
            self.calculate_descriptors()
        
        data = self.df[[self.RT_COLUMN] + descriptor_names]
        self.training_data, self.test_data = train_test_split(data, test_size=self.test_size, random_state=101)
    
    def get_training_data(self):
        if not self.data:
            self.build_dataset()
        
        return self.training_data
    
    def get_test_data(self):
        if not self.data:
            self.build_dataset()
        
        return self.test_data
