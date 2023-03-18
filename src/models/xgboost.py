from xgboost import XGBClassifier
from .base import Model

class Xgboost(Model):
    """
    A class that represents a xgboost model.

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
            Additional keyword arguments to pass to the DecisionTreeClassifier constructor.
        """
        self.model = XGBClassifier(**kwargs)
    
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
        self.model.fit(X, y)
    
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
    
    def __name__(self):
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the model.
        """
        return "xgboost"