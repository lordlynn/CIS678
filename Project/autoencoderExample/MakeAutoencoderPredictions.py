import matplotlib.pyplot as plt
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

    # Define the models
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This model maps latent space back to an image
    decoder = keras.Model(encoded, decoded)

setupModel()

# Load the model
autoencoder.load_weights('./model/50_epochs/autoencoder')
encoder.load_weights('./model/50_epochs/encoder')
decoder.load_weights('./model/50_epochs/decoder')

# Setup data
(x_train, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0

# Make prediction
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)



n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
plt.tight_layout()

for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.title("Decoded")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
pass