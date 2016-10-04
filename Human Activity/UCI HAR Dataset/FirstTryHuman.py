import numpy as np
from sklearn import neural_network, metrics,svm

Xtrain = np.loadtxt('C:/Users/Daniel/Google Drive/Programming/Datasets/Human Activity/UCI HAR Dataset/train/X_train.txt')
Ytrain = np.loadtxt('C:/Users/Daniel/Google Drive/Programming/Datasets/Human Activity/UCI HAR Dataset/train/y_train.txt')
Xtest = np.loadtxt('C:/Users/Daniel/Google Drive/Programming/Datasets/Human Activity/UCI HAR Dataset/test/X_test.txt')
Ytest = np.loadtxt('C:/Users/Daniel/Google Drive/Programming/Datasets/Human Activity/UCI HAR Dataset/test/y_test.txt')

print Xtrain.shape
print Xtest.shape

def model(Xtrain, Ytrain, Xtest, Ytest):
    NN.fit(Xtrain, Ytrain)

    predicted = NN.predict(Xtest)
    expected = Ytest

    print ("Classification Report for model %s:\n%s" % (NN, metrics.classification_report(expected, predicted)))
    print "Confusion Matrix:\n%s" % (metrics.confusion_matrix(expected, predicted))

NN = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='logistic')
model(Xtrain, Ytrain, Xtest, Ytest)

NN = neural_network.MLPClassifier(hidden_layer_sizes=282, activation='logistic')
model(Xtrain, Ytrain, Xtest, Ytest)
