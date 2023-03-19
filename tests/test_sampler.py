import sys
sys.path.append("../")
import importlib
import pandas as pd

def test_sampler() -> None:
    """Test Sampler class."""
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    target = pd.Series([7, 8, 9])
    sampler = getattr(importlib.import_module("src.sampler"), "Sampler")
    sampler = sampler(data, target)
    train_data, test_data, train_target, test_target = sampler.split()
    assert train_data.shape == (2, 2)
    assert test_data.shape == (1, 2)
    assert train_target.shape == (2,)
    assert test_target.shape == (1,)
    assert train_data.values.tolist() == [[1, 4], [2, 5]]
    assert test_data.values.tolist() == [[3, 6]]
    assert train_target.values.tolist() == [7, 8]
    assert test_target.values.tolist() == [9]