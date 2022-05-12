from flask import Flask

# setting path
import sys
sys.path.append('../mquery')
  
# importing
from mquery import pat_papertrade, screener, rnn_algotrade

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
