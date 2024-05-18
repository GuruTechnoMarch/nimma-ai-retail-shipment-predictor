from flask import Flask
from flask import request, jsonify
from src.shipment_predictor_optimized import ShipmentPredictorOptimized
import functions_framework

app = Flask(__name__)


model_path = 'models/large_xgbmodel.pkl'
scalar_path = 'models/large_scalarvalues.pkl'
processor = ShipmentPredictorOptimized(model_path=model_path, scalar_file=scalar_path)


@app.route('/predict', methods = ['POST'])
def run_request():
    input_requests = request.json
    if not isinstance(input_requests, list):
        return jsonify({'message': 'Malformed input, input should be a list of json'}), 400
    
    if len(input_requests) == 0:
        return jsonify([{'message': 'empty input'}]), 400
    
  
    return jsonify(
          [{"output": processor.process_request(input_request=input_request)} for input_request in  input_requests]
    )
        

@functions_framework.http
def run_prediction(request):
    input_requests = request.json
    if not isinstance(input_requests, list):
        return jsonify({'message': 'Malformed input, input should be a list of json'}), 400
    
    if len(input_requests) == 0:
        return jsonify([{'message': 'empty input'}]), 400
    
  
    return jsonify(
          [{"output": processor.process_request(input_request=input_request)} for input_request in  input_requests]
    )



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5010)