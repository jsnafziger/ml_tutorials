from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from timeit import default_timer as timer

model = ResNet50(weights='imagenet')

start = timer() # Time the inference
#img_path = 'elephant.jpg'
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
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
# Predicted: [('n02486410', 'baboon', 0.9025012), ('n02487347', 'macaque', 0.07545654), ('n02484975', 'guenon', 0.02052678)]