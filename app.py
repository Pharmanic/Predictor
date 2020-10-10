import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
import matplotlib.pyplot as plt
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
# model, forecast = pickle.load(open('model.pkl', 'rb'))

with open('model.pickle', 'rb') as f:
    supplies_test, supplies_forecast = pickle.load(f)

@app.route('/')
def home():
    output = supplies_forecast
    return render_template('index.html', prediction_text='adsfd  {}'.format(output))

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Order amount should be $ ()'.format(output))

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

if __name__ == "__main__":
    app.run(debug=True)