import keras
from keras import layers
from keras.datasets import mnist


encoder = None
decoder = None
autoencoder = None

def setupModel():
    global encoder, decoder, autoencoder

    # This is our input image
    input_img = keras.Input(shape=(28,28,1,))

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
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This model maps latent space back to an image
    decoder = keras.Model(encoded, decoded)


setupModel()


(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Training data shape (images, sizeX, sizeY): " + str(x_train.shape))
print("Test data shape (images, sizeX, sizeY): " + str(x_test.shape))


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Save the model
autoencoder.save_weights('./model/50_epochs/autoencoder')
encoder.save_weights('./model/50_epochs/encoder')
decoder.save_weights('./model/50_epochs/decoder')
