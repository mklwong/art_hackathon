#https://www.tensorflow.org/tutorials/generative/style_transfer

import numpy as np
import PIL.Image
import tensorflow as tf

class StyleContentModel(tf.keras.models.Model):
  """
  Somehow the style of the model has been embedded in the style_outputs, which comes from the preprocessed_input, which somehow
  doesn't require the style_image
  """
  def __init__(self, style_layers=None, content_layers=None):
    super(StyleContentModel, self).__init__()
    if content_layers is None:
        content_layers = ['block5_conv2'] 
    if style_layers is None:
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']
    self.vgg = self.vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def vgg_layers(self, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
  
    outputs = [vgg.get_layer(name).output for name in layer_names]
  
    model = tf.keras.Model([vgg.input], outputs)
    return model

  def gram_matrix(self,  input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [self.gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)



def make_loss(content_image, style_image, model):
  style_targets = model(style_image)['style']
  content_targets = model(content_image)['content']
  def style_content_loss(outputs):
      """
      The loss function takes as "style" and a "content" output and calculates the style and content loss from it.
      """
      style_weight=1e-2
      content_weight=1e4
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                             for name in style_outputs.keys()])
      style_loss *= style_weight / model.num_style_layers
  
      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                               for name in content_outputs.keys()])
      content_loss *= content_weight / len(model.content_layers)
      loss = style_loss + content_loss
      return loss
  return style_content_loss
  
def make_train(style_content_loss, model):
  @tf.function()
  def train_step(image):
    """
    The training step only takes in a "content" image, with a frozen style.
    """
    with tf.GradientTape() as tape:
      outputs = model(image)
      loss = style_content_loss(outputs)
  
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
  return train_step
  
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)