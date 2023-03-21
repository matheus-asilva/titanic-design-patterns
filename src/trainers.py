import importlib
import os
import pkgutil

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model: str, handle_missing: bool = True) -> None:
        """Initializes a Trainer object with the given parameters.

        Parameters:
        model : str
            The name of the machine learning model to be trained.
        handle_missing : bool
            Flag to set missing imputation. Fill missing values with 0.
        """
        self.model_name = model
        self._handles_missing = handle_missing

    def __get_model(self) -> None:
        """
        Private method that sets the self.model attribute to an instance of the
        specified machine learning model. Reads the available models from a
        specified directory, imports the module corresponding to the
        specified model, and creates an instance of the model.
        """
        src_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(src_path, "model")
        models = [
            model
            for _, model, _ in pkgutil.iter_modules([path])
            if model != "base"  # noqa: E501
        ]
        if self.model_name not in models:
            raise ValueError(
                f"{self.model_name} is not supported"
            )  # pragma: no cover # noqa: E501
        model_cls = importlib.import_module(f"model.{self.model_name}")
        self.model = getattr(
            model_cls, self.model_name.title().replace("_", "")
        )()  # noqa: E501


    def train(self, X, y) -> None:
        """
        Trains the specified machine learning model on the training data.

        Parameters:
        X : numpy.ndarray
            The feature matrix of the training dataset.
        y : numpy.ndarray
                The label vector of the training dataset.
        """
        self.__get_model()

        if not self.model._handles_missing:
            if not self._handles_missing:
                raise ValueError(f"{self.model_name} does not fit with missing information. Set '--handle_missing'.")
            else:
                self.model._handles_missing = self._handles_missing

        if self.model._handles_missing and not self._handles_missing:
            self.model._handles_missing = self._handles_missing

        if self.model._handles_missing:
            self.model.fit(X.fillna(0), y)
        else:
            self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Uses the trained model to make predictions on the prediction data
        and returns the predicted labels.

        Parameters:
        X : numpy.ndarray
            The feature matrix of the prediction dataset.
        Returns:
        numpy.ndarray
            The predicted labels for the prediction data.
        """
        if self.model._handles_missing:
            return self.model.predict(X.fillna(0))
        return self.model.predict(X)

    def optimize(self) -> None:
        """Optimizes the hyperparameters of the model."""
        self.model.optimize()

    def evaluate(self, X, y) -> float:
        """
        Evaluates the performance of the trained model on the
        prediction data using accuracy score.

        Parameters:
        X : numpy.ndarray
            The feature matrix of the evaluation dataset.
        y : numpy.ndarray
            The label vector of the evaluation dataset.
        Returns:
        float
            The accuracy score of the model on the test data.
        """
        pred = self.predict(X)
        return accuracy_score(pred, y)

    def save(self, path, model_name) -> None:
        """
        Saves the trained model to a specified path.

        Parameters:
        path : str
            The path to save the model.
        model_name : str
            The name to use when saving the model file.
        """
        joblib.dump(self.model, os.path.join(path, f"{model_name}.pkl"))
