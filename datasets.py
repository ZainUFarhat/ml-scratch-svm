# numpy
import numpy as np

# sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# datasets
class Datasets():

    """
    Description:
        Holds different classification datasets
    """

    # constructor
    def __init__(self, test_size, random_state):

        """
        Description:
            Constructor for our Datasets class
        
        Parameters:
            test_size: percentage of data to be allocated for testing
            random_state: random state chosen for reproducible output
        
        Returns:
            None
        """

        self.test_size = test_size
        self.random_state = random_state

    # blobs
    def make_blobs(self, num_samples, num_features):

        """
        Description:
            Loads toy dataset using sklearn make_blobs
        
        Parameters:
            num_samples: number of samples to generate for toy dataset
            num_features: number of features per training sample
        
        Returns:
            X, y, class_names, X_train, X_test, y_train, y_test
        """

        # make blobs of data
        X, y = datasets.make_blobs(n_samples = num_samples, n_features = num_features, 
                                                    centers = 2, cluster_std = 1.05, random_state = self.random_state)

        # class lables -1 for 0 and 1 for 1, this is for svm purposes
        y = np.where(y == 0, -1, 1)

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)    

        # class names
        class_names = ['positive', 'negative']

        # return
        return  X, y, class_names, X_train, X_test, y_train, y_test


    # breast cancer
    def load_breast_cancer(self):

        """
        Description:
            Loads sklearn's Breast Cancer Dataset

        Parameters:
            None
        
        Returns:
            X, y, feature_names, class_names, X_train, X_test, y_train, y_test
        """
        
        # load dataset
        data = datasets.load_breast_cancer()

        # load features, labels, and class names
        X, y, feature_names, class_names = data.data, data.target, data.feature_names, data.target_names

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # class lables -1 for 0 and 1 for 1, this is for svm purposes
        y = np.where(y == 0, -1, 1)
        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)

        # return
        return X, y, feature_names, class_names, X_train, X_test, y_train, y_test


    # diabetes
    def load_diabetes(self):

        """
        Description:
            Loads sklearn's Diabetes Dataset

        Parameters:
            None
        
        Returns:
            X, y_classification, feature_names, class_names, X_train, X_test, y_train, y_test
        """

        # This is a regression dataset but we will convert it to classification

        # load the dataset
        data = datasets.load_diabetes()

        # load features, "labels", and class names
        X, y, feature_names = data.data, data.target, data.feature_names

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # convert y to labels we want using median, if valua > median assign True, else False
        # we will use 1 (True) for has diabetes and 0 (False) for no diabetes
        y_classification = np.array([y > np.median(y)]).reshape(-1)   # (y > y.median()).astype(int)
        # class lables -1 for 0 and 1 for 1, this is for svm purposes
        y_classification = np.where(y_classification == 0, -1, 1)

        class_names = ['non-diabetic', 'diabetic']

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size = self.test_size, 
                                                                                                    random_state = self.random_state)

        # return
        return X, y_classification, feature_names, class_names, X_train, X_test, y_train, y_test