import flask
from flask import Flask, request, jsonify
from tensorflow.keras import backend
from PIL import Image
import numpy as np
import io
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.line_predictor import LinePredictor


# initialize our Flask application and the Keras model
# curl -X POST -F image=imagename 'http://localhost:8000/predict'
app = Flask(__name__)
predictor = None


def load_model():
    # load the pre-trained Keras model
    # Tensorflow bug: https://github.com/keras-team/keras/issues/2397
    with backend.get_session().graph.as_default() as _:
        predictor = LinePredictor()  # pylint: disable=invalid-name


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            # Tensorflow bug: https://github.com/keras-team/keras/issues/2397
            with backend.get_session().graph.as_default() as _:
                pred, conf = predictor.predict(image)
                print("METRIC confidence {}".format(conf))
                print("METRIC mean_intensity {}".format(image.mean()))
                print("INFO pred {}".format(pred))
            
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (p, c) in (pred, conf):
                r = {"prediction": p, "confidence": float(c)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0', port=8000, debug=False)
