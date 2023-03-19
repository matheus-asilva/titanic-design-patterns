from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from .base import Model


class Lightgbm(Model):
    """
    A class that represents a lightgbm model.

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
        Constructs a new LightGBM object.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword to the XGBClassifier constructor.
        """
        self.model = LGBMClassifier(**kwargs)
        self.params_grid = {
            "num_leaves": [25, 50, 75],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [5, 10],
            "n_estimators": [100, 200],
        }

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
        self : Lightgbm
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
        return "lightgbm"
