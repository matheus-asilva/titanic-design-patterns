from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from model.base import Model


class RandomForest(Model):
    """
    A class that represents a random forest model.

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
        Constructs a new RandomForest object.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to the RandomForestClassifier constructor.
        """
        self.model = RandomForestClassifier(**kwargs)
        self.params_grid = {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
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
        self : RandomForest
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
        return "random_forest"
