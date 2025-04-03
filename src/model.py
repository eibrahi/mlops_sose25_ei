import pandas as pd # type: ignore
import numpy as np
import pickle 
import openpyxl # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit # type: ignore
from sklearn.metrics import r2_score # type: ignore
from lightgbm import early_stopping


import os

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
    
    # speichern des Dataframes als excel
    combined_df.to_excel("combined_stock_data.xlsx", index=True)
    
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

def train_model(df):
    """Train LightGBM model with time series cross-validation"""
    print(f"Initial dataframe shape: {df.shape}")
    
    # Remove rows with NaN values (created by lag/window features)
    df = df.dropna()
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns 
                      if col not in ['return', 'stock', 'Close', 'Volume', 'Price'] 
                      and not pd.isna(df[col]).all()]  # Exclude columns that are all NaN
    
    # Create a copy of the data to avoid SettingWithCopyWarning
    X = df[feature_columns].copy()
    y = df['return'].copy()
    
    print(f"Number of features: {len(feature_columns)}")
    print("Feature columns:", feature_columns)
    
    # Convert all feature columns to float
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Print number of NaN values in each column
        nan_count = X[col].isna().sum()
        if nan_count > 0:
            print(f"Column {col} has {nan_count} NaN values")
    
    # Convert target to float
    y = pd.to_numeric(y, errors='coerce')
    
    # Remove any remaining NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    if len(X) == 0:
        raise ValueError("No samples remaining after data cleaning. Check your data for invalid values.")
    
    # Time series cross-validation
    n_splits = min(5, len(X) // 2)  # Ensure we don't have more splits than possible
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize model
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    
    # Train and evaluate
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=50)]
        )
        
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        scores.append(score)
    
    print(f"Average R2 score across folds: {np.mean(scores):.4f}")
    
    # Train final model on all data
    final_model = LGBMRegressor(
        n_estimators=model.best_iteration_,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    final_model.fit(X, y)
    
    return final_model, feature_columns

def main():
    # Load and prepare data
    stock_data_folder = "stock_data"
    df = prepare_data(stock_data_folder)
    
    # Train model
    model, feature_columns = train_model(df)
     
    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Save the model in the 'models' folder
    model_path = os.path.join("models", "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        print(f"Model saved at '{model_path}'")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    print("\nTop 10 most important features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))

if __name__ == "__main__":
    main()
