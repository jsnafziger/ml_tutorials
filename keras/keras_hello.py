# https://keras.io/applications/

# ----------------------
# ----------------------
# Added to deal with MITRE's Proxy
# Keras uses: from six.moves.urllib.request import urlretrieve
# https://pythonhosted.org/six/#module-six.moves.urllib.request
import urllib.request

#create the object, assign it to a variable
proxy = urllib.request.ProxyHandler({'https': 'gatekeeper-w.mitre.org:80'})
# construct a new opener using your proxy settings
opener = urllib.request.build_opener(proxy)
# install the opener on the module-level
urllib.request.install_opener(opener)
# ----------------------
# ----------------------

#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions

from keras.preprocessing import image
import numpy as np

#model = ResNet50(weights='imagenet')
model = MobileNet(weights='imagenet')

img_path = 'C:\\Users\\jnafziger\\Pictures\\elephant.jpg'
#img_path = 'C:\\Users\\jnafziger\\Pictures\\elephant_indian.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Need to figure out proxy settings for downloading within Keras