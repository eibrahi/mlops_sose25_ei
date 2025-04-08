import pickle # type: ignore
from datetime import datetime
import pandas as pd # type: ignore

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

def main():
    # Set the stock data folder
    stock_data_folder = "stock_data"
    
    try:
        # Make predictions
        predictions = predict_next_hour(stock_data_folder)
        
        # Print predictions
        print("\nPredictions for the next hour:")
        print(predictions)
        
        # Save predictions to Excel
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # predictions als csv speichern
        predictions.to_csv(f"predictions_{current_time}.csv")
        print(f"\nPredictions saved to predictions_{current_time}.csv")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
