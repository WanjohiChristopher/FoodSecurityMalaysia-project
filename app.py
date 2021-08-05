from flask import Flask, render_template, request
#from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@application.route('/')
def home():
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
    prediction = model.predict(feature_arr)
    output = round(prediction[0], 2)
    if output < 10:
        return render_template('crop.html', pred=' Dear farmer please consult an extension officer 🙏 {}'.format(output))
    else:
        return render_template('crop.html', pred='Dear Farmer according to the Details you gave us check on our Agricultural farm inputs store 🤗\n {}'.format(output))


'''@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]

    return jsonify(output)'''


@application.route('/about')
def crop_yield():
    return render_template('crop.html')


@application.route('/service')
def services():
    return render_template('services.html')


@application.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    application.run(debug=True)
