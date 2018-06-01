from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from timeit import default_timer as timer

model = VGG16(weights='imagenet')

start = timer() # Time the inference
img_path = '/home/john/Pictures/baboon.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
end = timer()
print('Seconds: ', end - start)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [('n02486410', 'baboon', 0.9025012), ('n02487347', 'macaque', 0.07545654), ('n02484975', 'guenon', 0.02052678)]