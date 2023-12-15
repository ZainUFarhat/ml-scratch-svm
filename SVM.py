# numpy
import numpy as np

class SVM():

    """
    Description:
         My from scratch implementation of the Support Vector Machine Algorithm
    """

    # constructor
    def __init__(self, epochs, lr, lambda_):

        """
        Description:
            Constuctor of our SVM class
        
        Parameters:
            epochs: number of iterations to train our model on
            lr: learning rate
            lambda_: lambda parameter to use for training optimization
    
        Returns:
            None
        """

        self.epochs = epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.w = []
        self.b = 0
    
    # fit
    def fit(self, X, y):

        """
        Description:
            Fits our SVM model
        
        Parameters:
            X: train features
            y: train labels
        
        Returns:
            None
        """

        # fetch number of features
        N, num_features = X.shape

        # make sure class labels are -1 and 1
        y = np.where(y <= 0, -1, 1)

        # intialize weight at radom between (0, 1)
        # self.w = np.random.uniform(0, 1, num_features)

        # initialize weights as zero
        self.w = np.zeros(num_features)

        # iterate over training set
        for _ in range(self.epochs):

            for (i, x_i) in enumerate(X):

                # check if this is upper or lower boundary
                condition = y[i] * (np.dot(x_i, self.w) - self.b) >= 1

                # condition
                if condition:
                    # update weights
                    self.w -= self.lr * (2 * self.lambda_ * self.w)
                
                else:
                    # update weights and bias
                    self.w -= self.lr * (2 * self.lambda_ * self.w - np.dot(x_i, y[i]))
                    self.b -= self.lr * y[i]
    
    # predict
    def predict(self, X):

        """
        Description:
            Predicts on our trained SVM model

        Parameters:
            X: test set
        
        Returns:
            predictions
        """

        # predictions
        predictions = np.sign(np.dot(X, self.w) - self.b)

        # return
        return predictions