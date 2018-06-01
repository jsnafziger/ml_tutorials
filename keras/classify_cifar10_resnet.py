from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from timeit import default_timer as timer

# Loading model that was trained and saved to local
model = load_model('/home/john/python/keras/saved_models/keras_cifar10_trained_model.h5')

img_path = '/home/john/Pictures/baboon.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
