from flask import Flask, request, jsonify
import util


app = Flask(__name__)

@app.route('/locations')
def locations():
    response = jsonify({
        'locations': util.locations()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/predict_price',methods=['POST'])
def predict_price():
    sqft = float(request.form['total_sqft'])
    location = float(request.form['location'])
    bhk = float(request.form['bhk'])
    bath = float(request.form['bath'])

    response = jsonify({
              'est_price':util.est_price(location, sqft,bhk , bath)

    })
    response.headers.add('Access-Control-Allow-Origin','*')

    return response

if __name__ == "__main__":
    print("starting python Flask server for price prediction...")
    util.load_saved()
    app.run()
