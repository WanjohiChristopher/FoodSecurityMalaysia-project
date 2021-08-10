from flask import Flask, render_template, request
#from flask import Flask, render_template, request
import pickle
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import cv2
import pickle
import ipywidgets as AppLayout
import ee
import geemap

application = Flask(__name__)
model = pickle.load(open('regressor.pkl', 'rb'))
model2 = pickle.load(open('finalized_model.pkl', 'rb'))


@application.route('/')
def home():
    #Map = geemap.Map()

    return render_template('index.html')


@application.route('/', methods=['POST'])
def predict():
    # print(request.form)
    # taking data from the form
    features = [int(x) for x in request.form.values()]
    #feature= request.form.to_dict()
    # feature=list(feature.values())
    #feature=list(map(int, feature)).reshape(1,-1)
    # keeping the features in an array
    feature_arr = [np.array(features)]
    # print(feature_arr)
    # performing prediction on our model
    #prediction = model.predict(feature_arr)
    prediction = model.predict(feature_arr)*-1000
    #output = round(prediction[0], 2)

    return render_template('crop.html', pred=' Dear farmer this is the expected yield  in hectogram per hectare HG/HA 🙏 {}'.format(prediction))


'''@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]

    return jsonify(output)'''


@application.route('/about')
def crop_yield():

    return render_template('crop.html')


@application.route('/cropdetection', methods=['POST'])
def cropdetection():

    # print(request.form)
    # taking data from the form

    feature = [float(y) for y in request.form.values()]
    # feature= request.form.to_dict()
    # feature=list(feature.values())
    # feature=list(map(int, feature)).reshape(1,-1)
    # keeping the features in an array
    feature_arrs = [np.array(feature)]
    # print(feature_arr)
    # performing prediction on our model
    # prediction = model.predict(feature_arr)
    predictions = model2.predict(feature_arrs)
    # output = round(prediction[0], 2)

    if predictions == 0:
        return render_template('detection.html', preds=' This is Oil palm 🙏 {}'.format(predictions))
    elif predictions == 1:
        render_template(
            'detection.html', preds=' This is Paddy  🌴 {}'.format(predictions))
    else:
        render_template(
            'detection.html', preds=' This is Rubber 🌾 {}'.format(predictions))

    return render_template('detection.html')


@application.route('/contact')
def detection():
    return render_template('detection.html')


if __name__ == '__main__':
    application.run(debug=True)
