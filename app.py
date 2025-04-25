from flask import Flask, request, jsonify
from demo import predict
from flask_cors import CORS

app = Flask(__name__)


CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json
        code1 = data['code1']
        code2 = data['code2']
        prediction, score = predict(code1, code2)  
        return jsonify({'prediction': int(prediction), 'similarity': score}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
