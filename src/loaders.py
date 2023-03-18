import pandas as pd


class CSVLoader:
    def __init__(self, path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
