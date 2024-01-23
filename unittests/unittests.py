import unittest
import pandas as pd
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "settings.json"

from training.train import DataProcessor, Training 


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)  # Check if DataFrame is not empty

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(50)  # Assuming there are at least 50 rows in the data
        self.assertEqual(df.shape[0], 50)
        self.assertEqual(set(df.columns), {'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target'})

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def setUp(self):
        self.dp = DataProcessor()
        self.tr = Training()

    def test_data_split(self):
        df = self.dp.data_extraction(self.train_path)
        X_train, X_test, y_train, y_test = self.tr.data_split(df, test_size=0.2)
        self.assertEqual(X_train.shape[0], int(0.8 * len(df)))
        self.assertEqual(X_test.shape[0], int(0.2 * len(df)))

    def test_train(self):
        df = self.dp.data_extraction(self.train_path)
        X_train, _, y_train, _ = self.tr.data_split(df, test_size=0.2)
        self.tr.train(X_train, y_train)
        self.assertIsNotNone(self.tr.model.tree_)

if __name__ == '__main__':
    unittest.main()