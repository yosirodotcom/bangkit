import os


from typing import Dict, Text
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mysql.connector
from flask import Flask, request, jsonify
import tempfile

# Get the current working directory
current_directory = os.getcwd()

# Specify the directory you want to join
specific_directory = 'model'

# Use os.path.join to create the complete path
full_path = os.path.join(current_directory, specific_directory)
print(full_path)

def predict(x):
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, full_path)

    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load(path)

    # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded([str(x)])
    tensor_data = titles[0][:5]
    python_list = [int(value.decode('utf-8')) for value in tensor_data.numpy()]
  return python_list

app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        try:
            # Get the value of 'x' from the JSON data in the request
            data = request.get_json()
            x = data.get('x')

            if x is None:
                return jsonify({"error": "'x' parameter is missing"})

            # Process the input
            result = predict(x)

            # Return the result as JSON
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
