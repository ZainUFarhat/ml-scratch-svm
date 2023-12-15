# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = np.equal(y_true, y_pred).sum()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return round(acc, 2) 

# scatter plot of given data
def scatter_plot(X, y, title, x_label, y_label, class_names, savepath):

    """
    Description:
        Plots a scatterplot based on X & y data provided

    Parameters:
        X: x-axis datapoints
        y: y-axis datapoints
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        class_names: names of our target classes
        savepath: path to save our scatterplot to

    Returns:
        None
    """

    # for plotting pruposes change -1 class back to 0
    y = np.where(y == -1, 0, 1)

    # intialize figure
    plt.figure(figsize = (7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # find features corresponding to class labels
    class_0, class_1 = X[y == 0], X[y == 1]

    # scatter plots of class features against themselves
    plt.scatter(class_0[:, 0], class_0[:, 1], label = class_names[0], c = 'r')
    plt.scatter(class_1[:, 0], class_1[:, 1], label = class_names[1], c = 'b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(savepath)

    # return
    return None

# helper function for plot_svm
def get_hyperplane_value(x, w, b, offset):

    """
    Description:
        Computes the hyperplane values to plot the three svm lines
    
    Parameters:
        x: features
        w: weights
        b: bias
        offset: offset parameter
    
    Returns:
        hyperplane_value
    """

    # hyperplane_value
    hyperplane_value = (-w[0] * x + b + offset) / w[1]

    # return
    return hyperplane_value

# plot
def plot_svm(model, X, y, title, x_label, y_label, class_names, savepath):

    """
    Description:
        Plots the hyperplanes of our fitted SVM model

    Parameters:
        model: fitted model
        X: features
        y: labels
        title: title of our plot
        x_label: x axis label 
        y_label: y axis label
        class_names: names of our classes
        savepath: path to save our plot to

    Returns:
        None
    """

    # intialize figure
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("lavender")


    # find features corresponding to class labels
    class_0, class_1 = X[y == -1], X[y == 1]

    # scatter plots of class features against themselves
    plt.scatter(class_0[:, 0], class_0[:, 1], label = class_names[0], c = 'r')
    plt.scatter(class_1[:, 0], class_1[:, 1], label = class_names[1], c = 'b')

    # min and max of x axis feature
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # y coordinates of first line
    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    # y coordinates of second line
    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

    # y coordinates of third line
    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

    # plot the three lines
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    # set limits for y axis
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    # title + axis
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()

    # save figure
    plt.savefig(savepath)