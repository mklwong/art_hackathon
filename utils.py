import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def get_file(file_or_url, output_name=None):
    if os.path.isfile(file_or_url):
        return file_or_url
    return tf.keras.utils.get_file(output_name, file_or_url)

def print_images(images, titles=None):
    print(len(images))
    if titles is not None:
        assert len(images) is len(titles)
    plt.figure(figsize=(12, 12))
    for ii, image in enumerate(images):
        plt.subplot(1, len(images), ii+1)
        plt.imshow(image[0])
        if titles is not None:
            plt.title(titles[ii])
    plt.show()

def img_scaler(image, max_dim = 512):

  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio, tf.int32)

  # Resizes the image based on the scaling constant generated above
  return tf.image.resize(image, new_shape)

def load_img(path_to_img, max_dim):

  # Reads and outputs the entire contents of the input filename.
  img = tf.io.read_file(path_to_img)

  # Detect whether an image is a BMP, GIF, JPEG, or PNG, and 
  # performs the appropriate operation to convert the input 
  # bytes string into a Tensor of type dtype
  img = tf.image.decode_image(img, channels=3)

  # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # Scale the image using the custom function we created
  img = img_scaler(img, max_dim)

  # Adds a fourth dimension to the Tensor because
  # the model requires a 4-dimensional Tensor
  return img[tf.newaxis, :]
