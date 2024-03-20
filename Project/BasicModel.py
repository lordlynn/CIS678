import prepareDataSet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


trainingImages, trainingTruth, testImages, testTruth = prepareDataSet.getDataset(flatten=True)

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



# Create an Support vector classifier model
clf = SVC(max_iter=1000, tol=0.001, cache_size=1000)

# Train the IsolationForest
clf.fit(trainingImages, trainingTruth)

# Make predictions on the test set
y_pred = clf.predict(testImages)

# Evaluate the accuracy
accuracy = accuracy_score(testTruth, y_pred)
print(f'SVC Test Accuracy: {accuracy}')

plt.figure()
plt.scatter(x, y_pred)
plt.scatter(x, testTruth)
plt.legend(["Prediction", "Expected"])
plt.xlabel("Sample (n)")
plt.ylabel("Classification (0 - normal, 1 - abnormal)")
plt.yticks([])
plt.ylim([-0.5, 1.5])
plt.title("SVC Test Accuracy")

plt.show()

pass