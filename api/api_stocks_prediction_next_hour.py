from flask import Flask, jsonify, request
import pandas as pd # type: ignore
from datetime import datetime
import pickle
# import os
from flask_cors import CORS  # type: ignore

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def predict_next_hour(stock_data_folder, model_path="models/model.pkl"):
    """Predict the next hour's returns for all stocks in the folder"""
    
    # Load the trained model and feature columns
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
    
    # Load the prepared csv file as combined_stocks_data
    df = pd.read_csv("combined_stock_data.csv", index_col=0, parse_dates=True)
        
    # Get the most recent data point for each stock
    latest_data = {}
    for stock in df['stock'].unique():
        stock_df = df[df['stock'] == stock].copy()
        
        # Get the most recent row with all features
        latest_features = stock_df[feature_columns].iloc[-1:].copy()
        
        try:
            # Make prediction
            predicted_return = model.predict(latest_features)[0]
            
            # Calculate predicted price
            last_close = stock_df['Close'].iloc[-1]
            predicted_price = last_close * (1 + predicted_return)
            
            # Get last timestamp
            last_timestamp = stock_df.index[-1]
            
            latest_data[stock] = {
                'Last Close': last_close,
                'Last Timestamp': last_timestamp,
                'Predicted Return': predicted_return,
                'Predicted Price': predicted_price,
            }
            
        except Exception as e:
            print(f"Error predicting for {stock}: {str(e)}")
            print(f"Features available: {latest_features.columns.tolist()}")
            print(f"Number of features: {len(latest_features.columns)}")
            print(f"Expected features: {feature_columns}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(latest_data, orient='index')
    
    # Format the results
    results_df['Predicted Return'] = results_df['Predicted Return'].map('{:.2%}'.format)
    results_df['Predicted Price'] = results_df['Predicted Price'].map('${:.2f}'.format)
    results_df['Last Close'] = results_df['Last Close'].map('${:.2f}'.format)
    
    # Sort by stock name
    results_df.sort_index(inplace=True)
    
    return results_df

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['GET'])
def get_predictions():
    """Endpoint to get predictions for all stocks"""
    try:
        # Get optional parameters from query string
        model_path = request.args.get('model_path', 'models/model.pkl')
        stock_data_folder = request.args.get('stock_data_folder', 'stock_data')

        # Get predictions using your existing function
        predictions_df = predict_next_hour(stock_data_folder, model_path)

        # Convert DataFrame to dictionary for JSON response
        predictions_dict = predictions_df.to_dict(orient='index')

        # Prepare response
        response = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions_dict
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/<stock_symbol>', methods=['GET'])
def get_prediction_for_stock(stock_symbol):
    """Endpoint to get prediction for a specific stock"""
    try:
        # Get optional parameters from query string
        model_path = request.args.get('model_path', 'models/model.pkl')
        stock_data_folder = request.args.get('stock_data_folder', 'stock_data')

        # Get predictions for all stocks
        predictions_df = predict_next_hour(stock_data_folder, model_path)

        # Check if stock exists
        if stock_symbol not in predictions_df.index:
            return jsonify({
                'error': f'Stock {stock_symbol} not found',
                'timestamp': datetime.now().isoformat()
            }), 404

        # Get prediction for specific stock
        stock_prediction = predictions_df.loc[stock_symbol].to_dict()

        # Prepare response
        response = {
            'timestamp': datetime.now().isoformat(),
            'stock': stock_symbol,
            'prediction': stock_prediction
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def start_server():
    """Function to start the Flask server"""
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    start_server()
