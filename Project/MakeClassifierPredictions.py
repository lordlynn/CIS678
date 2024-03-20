import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist

encoder = None
decoder = None
autoencoder = None

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



setupModel()

# Load the model
classifier.load_weights('./model/classifier/autoencoder')
# encoder.load_weights('./model/5/encoder')
# decoder.load_weights('./model/50_epochs/decoder')

# Setup data
(x_train, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0

# # Make prediction
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)



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