import json
import os
import subprocess
import sys

import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=["GET","POST"])
def index():
    print("hello world")
    return jsonify({"result": "done" })

app.run(host='0.0.0.0', port=5000)