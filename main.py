from search import *
from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def health_check():
    return "It works!"

@app.route("/search", methods = ["POST"])
def get_data():
    request_json = request.json
    headline_input = request_json.get('message')
    print("message: ", headline_input)
    result, response = predict_and_search(headline_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)