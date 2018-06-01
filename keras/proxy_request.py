import urllib.request

#create the object, assign it to a variable
proxy = urllib.request.ProxyHandler({'https': 'gatekeeper-w.mitre.org:80'})
# construct a new opener using your proxy settings
opener = urllib.request.build_opener(proxy)
# install the opener on the module-level
urllib.request.install_opener(opener)
# make a request
urllib.request.urlretrieve('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5', 'file.h5')