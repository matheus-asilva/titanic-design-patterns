import sys
sys.path.append("../")
import importlib
import pandas as pd

def test_csv_loader() -> None:
    loader = getattr(importlib.import_module("src.loaders"), "CSVLoader")
    loader = loader("./data/train.csv")
    data = loader.load()
    assert data.shape == (891, 12)