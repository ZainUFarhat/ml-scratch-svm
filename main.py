# datasets
from datasets import *

# SVM
from SVM import *

# utils
from utils import *

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Trains and tests our SVM Algorithm
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset global hyperparameters
    test_size = 0.2
    random_state = 42

    # dataset hyperparameters (blobs)
    dataset_name = 'Sklearn Blobs'
    num_samples = 150
    num_features = 2
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the sklearn blobs toy dataset
    X, y, class_names, X_train, X_test, y_train, y_test = datasets.make_blobs(num_samples = num_samples, num_features = num_features)

    print(f'Loading {dataset_name} Dataset...')
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nSVM\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # svm hyperparameters
    epochs = 1000
    lr = 0.01
    lambda_= 0.01 

    svm = SVM(epochs = epochs, lr = lr, lambda_ = lambda_)
    svm.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = svm.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')

    # scatter plot of original data
    title_scatter = f'{dataset_name} - Feature 1 vs. Feature 2'
    save_path_scatter = 'plots/blobs/blobs_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = 'Feature 1', y_label = 'Feature 2', 
                                class_names = class_names, savepath = save_path_scatter)
    
    title_boundary = f'{dataset_name} Decision Boundary - Feature 1 vs. Feature 2'  
    save_path_boundary = 'plots/blobs/blobs_decision_boundary.png'
    plot_svm(X = X[:, [0, 1]], y = y, model = svm, title = title_boundary, x_label = 'Feature 1',
                                y_label = 'Feature 2', class_names = class_names, savepath = save_path_boundary)

    print('Please refer to plots/blobs directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------\n')

    ######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Breast Cancer'

    # load the breast cancer dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nSVM\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # svm hyperparameters
    epochs = 1000
    lr = 0.1
    lambda_= 0.1 

    svm = SVM(epochs = epochs, lr = lr, lambda_ = lambda_)
    svm.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = svm.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    # scatter plot of original data
    title_scatter = f'{dataset_name} - {feature_names[0]} vs. {feature_names[1]}'
    save_path_scatter = 'plots/bc/bc_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = feature_names[0], y_label = feature_names[1], 
                                class_names = class_names, savepath = save_path_scatter)

    # decision boundary
    title_boundary = f'{dataset_name} Decision Boundary - {feature_names[0]} vs. {feature_names[1]}'  
    save_path_boundary = 'plots/bc/bc_decision_boundary.png'
    plot_svm(X = X[:, [0, 1]], y = y, model = svm, title = title_boundary, x_label = feature_names[0],
                                y_label = feature_names[1], class_names = class_names, savepath = save_path_boundary)

    print('Please refer to plots/bc directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------\n')

    #######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'

    # load the diabetes dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_diabetes()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\SVM\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # svm hyperparameters
    epochs = 100
    lr = 0.1
    lambda_= 0.1 

    svm = SVM(epochs = epochs, lr = lr, lambda_ = lambda_)
    svm.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = svm.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    # scatter plot of original data
    feature_1, feature_2 = 'ldl', 'hdl'
    title_scatter = f'{dataset_name} - {feature_1} vs. {feature_2}'
    save_path_scatter = 'plots/db/db_scatter.png'
    scatter_plot(X = X[:, [5, 6]], y = y, title = title_scatter, x_label = feature_1, y_label = feature_2, 
                                class_names = class_names, savepath = save_path_scatter)


    # decision boundary
    title_boundary = f'{dataset_name} Decision Boundary - {feature_1} vs. {feature_2}'  
    save_path_boundary = 'plots/db/db_decision_boundary.png'
    plot_svm(X = X[:, [5, 6]], y = y, model = svm, title = title_boundary, x_label = feature_1,
                                y_label = feature_2, class_names = class_names, savepath = save_path_boundary)
    
    print('Please refer to plots/db directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------')
    
    # return
    return None

if __name__ == '__main__':

    # run everything
    main()