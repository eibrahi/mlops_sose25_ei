import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

def prepare_data(stock_data_folder):
    """Load and prepare data from all stock files"""
    all_stocks_data = []
    
    for file in os.listdir(stock_data_folder):
        if file.endswith('.csv'):
            # Load data
            df = pd.read_csv(os.path.join(stock_data_folder, file))
            
            # Check if 'Date' is a column or index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                # If Date is already the index
                df.index = pd.to_datetime(df.index)
            
            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create features
            df = create_features(df)
            
            # Add stock identifier
            stock_name = file.split('_')[0].upper()
            df['stock'] = stock_name
            
            all_stocks_data.append(df)
    
    # Combine all stocks
    combined_df = pd.concat(all_stocks_data)
    combined_df.sort_index(inplace=True)
    
    return combined_df

def create_features(df):
    """Create lag and window features from price data"""
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate returns (with explicit fill_method=None to address the warning)
    df['return'] = df['Close'].pct_change(fill_method=None)
    
    # Create lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
    # Create window features
    for window in [5, 10, 20]:
        # Price windows
        df[f'close_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'close_std_{window}'] = df['Close'].rolling(window=window).std()
        
        # Return windows
        df[f'return_mean_{window}'] = df['return'].rolling(window=window).mean()
        df[f'return_std_{window}'] = df['return'].rolling(window=window).std()
        
    # Volume features
    df['volume_lag_1'] = df['Volume'].shift(1)
    df['volume_mean_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_std_5'] = df['Volume'].rolling(window=5).std()
    
    return df

def predict_next_hour(stock_data_folder, model_path="models/model.pkl"):
    """Predict the next hour's returns for all stocks in the folder"""
    
    # Load the trained model and feature columns
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
    
    # Load and prepare the most recent data
    df = prepare_data(stock_data_folder)
    
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
        predictions.to_excel(f"predictions_{current_time}.xlsx")
        print(f"\nPredictions saved to predictions_{current_time}.xlsx")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
