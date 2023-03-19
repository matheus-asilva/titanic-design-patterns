import sys
sys.path.append("../")
import importlib
import pandas as pd


trainer = getattr(importlib.import_module("src.trainers"), "Trainer")

# def test_get_model() -> None:
#     """Test get_model method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     assert trainer.model.__class__.__name__ == "DecisionTree"

# def test_train() -> None:
#     """Test train method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     trainer.train(pd.DataFrame(), pd.Series())
#     assert trainer.model.__class__.__name__ == "DecisionTree"

# def test_evaluate() -> None:
#     """Test evaluate method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     trainer.train(pd.DataFrame(), pd.Series())
#     assert trainer.evaluate(pd.DataFrame(), pd.Series()) == 1.0

# def test_save() -> None:
#     """Test save method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     trainer.train(pd.DataFrame(), pd.Series())
#     trainer.save("./models", "decision_tree")
#     assert trainer.model.__class__.__name__ == "DecisionTree"

# def test_optimize() -> None:
#     """Test optimize method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     trainer.train(pd.DataFrame(), pd.Series())
#     trainer.optimize()
#     assert trainer.model.__class__.__name__ == "DecisionTree"

# def test_predict() -> None:
#     """Test predict method."""
#     trainer = TRAINER("decision_tree")
#     trainer._Trainer__get_model()
#     trainer.train(pd.DataFrame(), pd.Series())
#     assert trainer.predict(pd.DataFrame()) == 1.0