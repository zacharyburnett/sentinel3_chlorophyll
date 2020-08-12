from functools import lru_cache

import numpy
import pandas
from torch.utils.data.dataset import Dataset


class TextFileDataset(Dataset):
    def __init__(self, filename: str, separator: str = None):
        if separator is None:
            separator = ','
        self.filename = filename
        self.data_frame = pandas.read_csv(self.filename, sep=separator)

    @lru_cache(maxsize=1)
    def __len__(self):
        with open(self.filename) as file:
            return len(file.readlines())

    def __getitem__(self, index: int) -> numpy.array:
        return self.data_frame[index]

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.filename})'
