from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from model.base import Model


class DecisionTree(Model):  # pragma: no cover
    """
    A class that represents a decision tree model.

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
        Constructs a new DecisionTree object.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to the DecisionTreeClassifier constructor.
        """
        self.model = DecisionTreeClassifier(**kwargs)
        self.params_grid = {
            "max_depth": [2, 3, 5, 10, 20],
            "min_samples_leaf": [5, 10, 20, 50, 100],
            "criterion": ["gini", "entropy"],
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
        self : DecisionTree
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
        return "decision_tree"
