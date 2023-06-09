import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import importlib
import pandas as pd
from sklearn.datasets import make_classification

TRAINER = getattr(importlib.import_module("src.trainers"), "Trainer")
N_FEATURES = 5
N_SAMPLES = 500
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=2, random_state=2023, shuffle=False)
X = pd.DataFrame(X, columns=[f"FEAT{num}" for num in range(1, N_FEATURES + 1)])
y = pd.Series(y)



# write your tests here
def test_trainer() -> None:
    """Test Trainer class."""
    trainer = TRAINER("decision_tree")
    assert trainer.model_name == "decision_tree"

def test_save() -> None:
    """Test save method."""
    trainer = TRAINER("decision_tree")
    trainer.train(X, y)
    trainer.save("./models", "decision_tree_test")
    assert os.path.exists("./models/decision_tree_test.pkl")
    os.remove("./models/decision_tree_test.pkl")

def test_get_model() -> None:
    """Test get_model method."""
    trainer = TRAINER("decision_tree")
    trainer._Trainer__get_model()
    assert trainer.model.__class__.__name__ == "DecisionTree"

def test_train() -> None:
    """Test train method."""
    trainer = TRAINER("decision_tree")
    trainer._Trainer__get_model()
    trainer.train(X, y)
    assert trainer.model.__class__.__name__ == "DecisionTree"

def test_evaluate() -> None:
    """Test evaluate method."""
    trainer = TRAINER("decision_tree")
    trainer._Trainer__get_model()
    trainer.train(X, y)
    assert trainer.evaluate(X, y) == 1.0

def test_optimize() -> None:
    """Test optimize method."""
    trainer = TRAINER("decision_tree")
    trainer._Trainer__get_model()
    trainer.train(X, y)
    trainer.optimize()
    assert trainer.model.__class__.__name__ == "DecisionTree"

def test_predict() -> None:
    """Test predict method."""
    trainer = TRAINER("decision_tree")
    trainer._Trainer__get_model()
    trainer.train(X, y)
    assert trainer.predict(X).mean() == 0.5
