import prepareDataSet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


trainingImages, trainingTruth, testImages, testTruth = prepareDataSet.getDataset()

# Ideas for improvements: 
#   1.) Normalize the images (make sure all images use full 0-255 range for brightness)
#   2.) Can use non destructive transforms to augment data 



# Create a Random Forest classifier (approaches 0.5 random chance??)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(trainingImages, trainingTruth)

# Make predictions on the test set
y_pred = clf.predict(testImages)

# Evaluate the accuracy
accuracy = accuracy_score(testTruth, y_pred)
print(f'Random Forest Test Accuracy: {accuracy}')

x = [i for i in range(len(testTruth))]
plt.figure()
plt.scatter(x, y_pred)
plt.scatter(x, testTruth)
plt.legend(["Prediction", "Expected"])
plt.xlabel("Sample (n)")
plt.ylabel("Classification (0 - normal, 1 - abnormal)")
plt.yticks([])
plt.ylim([-0.5, 1.5])
plt.title("Random Forest Test Accuracy")



# Create an IsolationForest model
clf = IsolationForest(contamination=0.5, random_state=42)

# Train the IsolationForest
clf.fit(trainingImages)

# Make predictions on the test set
y_pred = clf.predict(testImages)

# Convert the predictions (-1 for outliers, 1 for inliers) to binary labels (0 or 1)
y_pred = (y_pred + 1) // 2

accuracy = accuracy_score(testTruth, y_pred)
print(f'Isolation Forest Test Accuracy: {accuracy}')

plt.figure()
plt.scatter(x, y_pred)
plt.scatter(x, testTruth)
plt.legend(["Prediction", "Expected"])
plt.xlabel("Sample (n)")
plt.ylabel("Classification (0 - normal, 1 - abnormal)")
plt.yticks([])
plt.ylim([-0.5, 1.5])
plt.title("Isolation Forest Test Accuracy")

plt.show()

pass