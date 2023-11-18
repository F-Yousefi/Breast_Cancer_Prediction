import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


class Dataset:
    def __init__(self, path):
        self.__dataset = pd.read_csv(path)
        self.shape = self.__dataset.shape

    def dropna(self): self.__dataset.dropna(axis=1, inplace=True)

    def to_numerical(self):
        self.__dataset = self.__dataset.drop(columns=["id"])
        le = preprocessing.LabelEncoder()
        le.fit(self.__dataset.diagnosis)
        self.__dataset['diagnosis'] = le.transform(self.__dataset.diagnosis)

    def generate(self): return (
        self.__dataset[self.__dataset.columns[1:]], self.__dataset[self.__dataset.columns[0]])
