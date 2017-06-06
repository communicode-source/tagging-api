# app.py
# Communicode-tagify
# Author : Abraham Oliver <abeoliver.116@gmail.com>
# Date : 5/25/2017

# Import flask microframework
from flask import Flask
# Import Jsonify, a tool for converting dictionaries into JSON files
from flask import jsonify
# Import various Flask tools
from flask import request, make_response, abort
# Import tool for importing models
from nn import getModel
# Os for filepaths
import os

# Create app
app = Flask(__name__)

# Get predicted tags for a given request
@app.route('/tagify/v1.0/tagify/<modelName>', methods = ["POST"])
def getTags(modelName):
    # Kill if JSON not given
    if not request.json: abort(400)
    # Get directory
    dir = os.path.dirname(__file__)
    # Get desired model
    filename = dir + "/models/{0}".format(modelName)
    print(filename)
    model = getModel(filename)
    # Calculate tags and confidences
    tags = model.tag(request.json["description"])
    # Return as JSON
    return jsonify(tags)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# Run app
if __name__ == "__main__":
    app.run(debug = True)
