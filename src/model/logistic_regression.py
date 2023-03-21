from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
import numpy as np

from model.base import Model


class LogisticRegression(Model):  # pragma: no cover
    """
    A class that represents a logistic regression model.

    ...

    Methods
    -------
    fit(X, y)
        Fit the model according to the given training data.
    __name__()
        Returns the name of the model.
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructs a new LogisticRegression object.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to the LogisticRegression constructor.
        """
        self.model = LogisticRegressionClassifier(**kwargs)
        self.params_grid = {
            'C' : [100, 10, 1.0, 0.1, 0.01],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
            'max_iter' : [100, 1000,2500, 5000]

        }
        self._handles_missing = False

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : LogisticRegression
            Returns the instance itself.
        """
        self.X = X
        self.y = y

        self.model.fit(self.X, self.y)

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        return self.model.predict(X)

    def optimize(self):
        """
        Optimizes the hyperparameters of the model.
        """
        tuned_model = GridSearchCV(
            self.model, param_grid=self.params_grid, cv=5, scoring="accuracy"
        )
        tuned_model.fit(self.X, self.y)
        self.model = tuned_model.best_estimator_

    def __name__(self):
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the model.
        """
        return "logistic_regression"
