import numpy as np
import base64

from flask import Flask, request, render_template, make_response
import keras
from io import BytesIO
from skimage import io as skio
from skimage.transform import resize
from utils import make_mnist

app = Flask(__name__, static_url_path='/static')
clf = keras.models.load_model('clf.h5py')

@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/recognizer', methods=['POST'])
def recognize():
    data = request.get_json(silent=True)['image']
    data = data[22:]

    img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]

    img = make_mnist(img)
    img = img.reshape(1,28,28,1)

    number = clf.predict_classes(img)[0]

    return make_response(str(number),200)
