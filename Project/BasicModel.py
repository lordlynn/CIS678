import prepareDataSet
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


trainingImages, trainingTruth, testImages, testTruth = prepareDataSet.getDataset(flatten=True)


def compressData():
    global trainingImages, testImages

    pca = PCA(n_components=500)
    pca.fit(trainingImages)
    EVR = pca.explained_variance_ratio_

    print("Proportion of the varaince explained by the components: " + str(np.sum(EVR)))
    
    plt.figure()
    plt.plot(EVR)
    plt.xlabel("Component")
    plt.ylabel("Explained Variance")
    plt.title("PCA: Explained Varaince Ratios")
    
    trainingImages = pca.transform(trainingImages)
    testImages = pca.transform(testImages)



def plotAccuracy(prediction, truth):
    numPoints = len(truth)
    x = [i for i in range(numPoints)]
    s1 = [175 for i in range(numPoints)]
    s2 = [175 for i in range(numPoints)]

    plt.figure()
    plt.scatter(x, prediction, s=s1, marker="|")
    plt.scatter(x, truth, s=s2, marker="_")
    plt.legend(["Prediction", "Expected"])
    plt.xlabel("Sample (n)")
    plt.ylabel("Classification (0 - normal, 1 - abnormal)")
    plt.yticks([])
    plt.ylim([-0.5, 1.5])


def randomForest(hyperParameters):
    # Use random search to find the best hyperparameters
    rf = RandomForestClassifier()

    # Do a grid search of the parameters with 4-fold corss validation. Use the best result
    clf = GridSearchCV(rf, param_grid=hyperParameters, cv=4, n_jobs=8)

    # Train the classifier
    clf.fit(trainingImages, trainingTruth)

    # Make predictions on the test set
    y_pred = clf.predict(testImages)

    # Evaluate the accuracy
    accuracy = accuracy_score(testTruth, y_pred)
    print(f'Random Forest Test Accuracy: {accuracy}')
    
    plotAccuracy(y_pred, testTruth)
    plt.title("Random Forest Test Accuracy")


   
def svc(hyperParameters):
    # Create an Support vector classifier model
    sv = SVC(kernel="rbf")
    clf = GridSearchCV(sv, param_grid=hyperParameters, cv=4, n_jobs=8)

    # Train the support vector classifier
    clf.fit(trainingImages, trainingTruth)

    # Make predictions on the test set
    y_pred = clf.predict(testImages)

    # Evaluate the accuracy
    accuracy = accuracy_score(testTruth, y_pred)
    print(f'SVC Test Accuracy: {accuracy}')

    plotAccuracy(y_pred, testTruth)
    plt.title("SVC Test Accuracy")


# Use PCA to compress the data
compressData()

# Random forest 
hyperParameters = {"n_estimators": [50, 100, 250],
                   "max_depth":    [10, 50, 100, 250]}

randomForest(hyperParameters)

# Support vector classifier
hyperParameters = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200],
                   "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200]}

svc(hyperParameters)

# Block at end of program to show the plots
plt.show()

