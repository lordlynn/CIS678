import keras
from keras import layers
import prepareDataSet
import numpy as np


encoder = None
decoder = None
classifier = None

def setupModel():
    global encoder, decoder, classifier

    # This is our input image
    input_img = keras.Input(shape=(175, 167, 1,))

    # Encoder reduces image to latent space
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # latent space representation of image with shape (4, 4, 8)

    # Decoder recreates image from latent space
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    # Output a values between 0 and 1
    x = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Flatten the output and add a dense layer to produce a single value
    x = layers.Flatten()(x)
    decoded = layers.Dense(1, activation='sigmoid')(x)
    
    # This model maps an input to its classification 
    classifier = keras.Model(input_img, decoded)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                       metrics=['accuracy']) # TODO: investigate which metrics to display here, this is just a placeholder




trainingImages, trainingTruth, testImages, testTruth = prepareDataSet.getDataset()
imgShape = trainingImages[0].shape

print("Training data shape: " + str(trainingImages[0].shape))
print("Number of training data: " + str(len(trainingImages)))

setupModel()


trainingImages = np.array(trainingImages)
trainingTruth = np.array(trainingTruth)
testImages = np.array(testImages)
testTruth = np.array(testTruth)

classifier.fit(trainingImages, trainingTruth,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(testImages, testTruth))

# Save the model
classifier.save_weights('./model/classifier/10_epochs')

