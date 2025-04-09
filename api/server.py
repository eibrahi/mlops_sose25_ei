from flask import Flask, jsonify
import pickle
from datetime import datetime
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Lade das Model beim Start der Anwendung
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    MODEL = model_data['model']
    FEATURE_COLUMNS = model_data['feature_columns']

def format_predictions(predictions_df):
    """Format predictions for JSON response"""
    return {
        stock: {
            'last_close': str(row['Last Close']),
            'last_timestamp': str(row['Last Timestamp']),
            'predicted_return': str(row['Predicted Return']),
            'predicted_price': str(row['Predicted Price'])
        }
        for stock, row in predictions_df.iterrows()
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint für Health Check"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/v1/predictions', methods=['GET'])
def get_predictions():
    """Endpoint für Vorhersagen"""
    try:
        # Load data
        df = pd.read_csv("combined_stock_data.csv", index_col=0, parse_dates=True)
        
        latest_data = {}
        for stock in df['stock'].unique():
            stock_df = df[df['stock'] == stock].copy()
            latest_features = stock_df[FEATURE_COLUMNS].iloc[-1:].copy()
            
            try:
                predicted_return = MODEL.predict(latest_features)[0]
                last_close = stock_df['Close'].iloc[-1]
                predicted_price = last_close * (1 + predicted_return)
                last_timestamp = stock_df.index[-1]
                
                latest_data[stock] = {
                    'Last Close': last_close,
                    'Last Timestamp': last_timestamp,
                    'Predicted Return': predicted_return,
                    'Predicted Price': predicted_price,
                }
                
            except Exception as e:
                app.logger.error(f"Error predicting for {stock}: {str(e)}")
                continue
        
        results_df = pd.DataFrame.from_dict(latest_data, orient='index')
        
        # Format results
        results_df['Predicted Return'] = results_df['Predicted Return'].map('{:.2%}'.format)
        results_df['Predicted Price'] = results_df['Predicted Price'].map('${:.2f}'.format)
        results_df['Last Close'] = results_df['Last Close'].map('${:.2f}'.format)
        results_df.sort_index(inplace=True)
        
        formatted_predictions = format_predictions(results_df)
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'predictions': formatted_predictions
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
