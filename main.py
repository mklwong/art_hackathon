#python = 3.7
#anaconda = 5.0.0
#tensorflow = 2.0.0

import utils as _utils
import model as _model
import IPython.display as display
from matplotlib import pyplot as plt
import progressbar
import tensorflow as tf

def print_images(images, titles=None):
    if titles is not None:
        assert len(images) is len(titles)
    plt.figure(figsize=(12, 12))
    for ii, image in enumerate(images):
        plt.subplot(1, len(images), ii+1)
        plt.imshow(image[0])
        if titles is not None:
            plt.title(titles[ii])
    plt.show()

def load_file_or_url(file_or_url, output_name=None, max_dim=512):
    """
    load_file_or_url(file_or_url, output_name=None, max_dim=512):
    
    
    """
    return _utils.load_img(_utils.get_file(file_or_url, output_name), max_dim)

def transfer_style(content_image, style_image, model=_utils.hub_model):
    """
    transfer_tyle(content_image, style_image, model=hub_model)
    
    This function transfers a based image (content_image) into the style of a style image (style_image)
    given a style transfer model (model).
    
    content_image: EagerTensor
      A image file generated using the load_file_or_url function that will be transformed.
    style_image: EagerTensor
      A image file generated using the load_file_or_url function that it used to define the style.
    model: Tensorflow model, default 
      A tensorflow style transform model that takes in two EagerTensors are input arguments.
    """
    return _utils.hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
def run_demo():
    """
    Runs a demo of the code using 
    """
    content_image = load_file_or_url('https://images.unsplash.com/photo-1501820488136-72669149e0d4', 'photo-1501820488136-72669149e0d4')
    style_image = load_file_or_url('https://upload.wikimedia.org/wikipedia/commons/8/8c/Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg','Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg')
    output_image = transfer_style(content_image, style_image)
    print_images([content_image, style_image, output_image], ['Source image','Style Image', 'Output Image'])

def run_manual_demo(n=100):
    """
    Runs a demo of the code using 
    """
    content_image = load_file_or_url('https://images.unsplash.com/photo-1501820488136-72669149e0d4', 'photo-1501820488136-72669149e0d4')
    style_image = load_file_or_url('https://upload.wikimedia.org/wikipedia/commons/8/8c/Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg','Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg')
    extractor, train_step, output_image = run_model_train(style_image, content_image=content_image)
    with progressbar.ProgressBar(max_value=n) as bar: 
        for ii in range(n):
            bar.update(ii)
            train_step(output_image)
    print_images([content_image, style_image, output_image], ['Source image','Style Image', 'Output Image'])
    
def run_model_train(style_image, extractor=None, content_image=None):
    if extractor is None:
        extractor = _model.StyleContentModel()
    if content_image is None:
        content_image = style_image*0
    image = _model.tf.Variable(content_image)
    train_step = _model.make_train(content_image, style_image, extractor)
    return extractor, train_step, image