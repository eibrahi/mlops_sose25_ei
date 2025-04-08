import pandas as pd # type: ignore
import numpy as np
import pickle 
#import openpyxl # type: ignore
#from sklearn.preprocessing import StandardScaler # type: ignore
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit # type: ignore
from sklearn.metrics import r2_score # type: ignore
from lightgbm import early_stopping


import os

def prepare_data(stock_data_folder):
    """Load and prepare data from all stock files"""
    print(f"\n=== Starte prepare_data mit Ordner: {stock_data_folder} ===\n")
    
    all_stocks_data = []
    print(f"Initialisiere leere Liste all_stocks_data: {all_stocks_data}")
    
    files = os.listdir(stock_data_folder)
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"Gefundene CSV-Dateien ({len(csv_files)}): {csv_files}")
    
    for i, file in enumerate(csv_files):
        print(f"\n--- Verarbeite Datei {i+1}/{len(csv_files)}: {file} ---")
        file_path = os.path.join(stock_data_folder, file)
        
        # Hier ist die geänderte Importlogik
        df = pd.read_csv(file_path)
        
        # Überprüfen Sie, ob 'Datetime' in den Spalten ist
        if 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'], format='ISO8601')
            df.set_index('Date', inplace=True)
            df.drop('Datetime', axis=1, errors='ignore', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
            df.set_index('Date', inplace=True)
            
        print(f"Datums-Bereich: {df.index.min()} bis {df.index.max()}")
        
        # Rest Ihrer Funktion bleibt gleich...
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = create_features(df)
        stock_name = file.split('_')[0].upper()
        df['stock'] = stock_name
        all_stocks_data.append(df)
    
    # Kombinieren der DataFrames
    combined_df = pd.concat(all_stocks_data)
    combined_df.sort_index(inplace=True)
    
    # Speichern des DataFrames als csv
    combined_df.to_csv("combined_stock_data.csv", index=True)
    
    print(f"\n=== Zusammenfassung des kombinierten DataFrames ===")
    print(f"Zeitraum: {combined_df.index.min()} bis {combined_df.index.max()}")
    
    
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

    # Erstelle ein Dictionary mit Model und Feature-Spalten
    model_data = {
        'model': model,
        'feature_columns': feature_columns
    }

    # Save the model data in the 'models' folder
    model_path = os.path.join("models", "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        print(f"Model and feature columns saved at '{model_path}'")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    print("\nTop 10 most important features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))

if __name__ == "__main__":
    main()
