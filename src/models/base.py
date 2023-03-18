from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        """
        Trains the machine learning model on the given data.
        
        Parameters:
        X : array-like
            The feature matrix of the training data.
        y : array-like
            The target vector of the training data.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Uses the trained machine learning model to make predictions on new data.
        
        Parameters:
        X : array-like
            The feature matrix of the data to be predicted.
            
        Returns:
        array-like
            The predicted target vector.
        """
        pass