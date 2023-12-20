# model.py

# Import the necessary libraries from scikit-learn for loading a dataset, 
# splitting it into training and testing sets, and creating a RandomForestClassifier.

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
#Load the Iris dataset from scikit-learn and split it into training and testing sets.
# This line uses scikit-learn's datasets module to load the Iris dataset. The Iris dataset 
# is a well-known dataset in machine learning and consists of 150 samples of iris flowers, 
# each belonging to one of three species (setosa, versicolor, or virginica). Each sample has four 
# features (sepal length, sepal width, petal length, and petal width).


iris = datasets.load_iris()

# This line uses the train_test_split function from scikit-learn to split the dataset into 
# training and testing sets. This function takes the features (iris.data) and labels (iris.target) 
# as input and returns four sets: training features (X_train), testing features (X_test), training 
# labels (y_train), and testing labels (y_test).

# test_size=0.2: This parameter specifies that 20% of the data should be reserved for testing, and 
# the remaining 80% will be used for training.

# random_state=42: This parameter sets the random seed for reproducibility. Setting a random seed ensures that the data is split in the same way each time the script is run, which is useful for reproducibility in machine learning experiments.

# The resulting sets (X_train, X_test, y_train, y_test) can be used to train a machine learning 
# model on the training data and evaluate its performance on the testing data.
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a model
#Create a RandomForestClassifier with 100 trees and train it on the training data.

#This line creates an instance of the RandomForestClassifier class from scikit-learn. 
# n_estimators=100 specifies that the random forest should consist of 100 decision trees. 
# The RandomForestClassifier is an ensemble learning method that builds a collection of decision 
# trees and merges their predictions to improve overall accuracy and control overfitting.


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# After these two lines of code, the model variable now holds a trained RandomForestClassifier, 
# and you can use this model to make predictions on new, unseen data. For example, as shown in 
# the previous code snippets, you can use this model to predict the class of a new set of features:

# prediction = model.predict(features)
