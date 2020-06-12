from __future__ import division, print_function
#from cloudant import Cloudant
from flask import Flask,request, render_template
#import json
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from gevent.pywsgi import WSGIServer

#import sys
import os
#import glob
global model,graph
#import tensorflow as tf
#graph = tf.get_default_graph()
app = Flask(__name__)

#MODEL_PATH = 'cnn_catdog.h5'
from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
import pickle
app = Flask(__name__)
import keras
import tensorflow as tf
model = keras.models.load_model(r"E:\ML PROJ SB\toxicmodel.h5")
model._make_predict_function()
graph = tf.get_default_graph()
#model = load("toxic.h5")
#model = load_model("toxicmodel.h5")



def model_predict(f, model):
    loaded_vec =CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("toxic_cmnt.pkl", "rb")))
    f= f.split("delimiter")
    g=loaded_vec.transform(f)
    with graph.as_default():
	    result = model.predict(g)
    result = (result>0.5)
    #result=model.predict(loaded_vec.transform(f))
   # result = (result>0.5)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('bse.html')


@app.route('/analyze', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.form['comment']
        preds = model_predict(f, model)
        ls=["Toxic","Severe_toxic","Obscene","Threat","Insult","Identify hate"]
        result = "The comment is: "
        for x in preds:
            cnt=0
            fls=0
            print(x)
            for y in x:
                print(y)
                if(y == True):
                    result+=str(ls[cnt])
                    result+=" "
                    fls+=1
                cnt = cnt+1
            if fls==0:
               result+="good" 
        return render_template('bse.html', prediction_text='{}'.format(result))
    return None


if __name__ == '__main__':
    app.run(debug=True)
     # port = int(os.getenv('PORT', 8000))
     #app.run(host='0.0.0.0', port=port, debug=True)
     # http_server = WSGIServer(('0.0.0.0', port), app)
     # http_server.serve_forever()
     #app.run(debug=True)