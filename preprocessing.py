import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def wrangling(df):
 
    df = df.drop(columns = ['quote_asset_volume', 'num_trades', 'taker_buy_base_volume','taker_buy_quote_volume', 'ignore'])

    df.sort_values("open_time", inplace=True)
    df.sort_values("open_time", inplace=True)

    df = df.drop(columns = ['open_time', 'close_time'])

    df['target'] = df['close'].shift(-1)

    df.dropna(inplace = True)
    
    return df
    
    

def create_sequences(X, y, window_size):
    
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)


def prepare_data(
        df,
        feature_cols = ["open", "high", "low", "close","volume"],
        window_size=30,
        train_ratio=0.8):
            

            # Train / validation split
            train_size = int(len(df) * train_ratio)
            train_df = df.iloc[:train_size]
            val_df   = df.iloc[train_size:]

            # Scaling
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train = scaler_X.fit_transform(train_df[feature_cols])
            X_val   = scaler_X.transform(val_df[feature_cols])

            y_train = scaler_y.fit_transform(train_df[['target']])
            y_val   = scaler_y.transform(val_df[['target']])

            # Sequence creation
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
            X_val_seq, y_val_seq     = create_sequences(X_val, y_val, window_size)

            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

            return (
                X_train_tensor,
                y_train_tensor,
                X_val_tensor,
                y_val_tensor,
                scaler_X,
                scaler_y
                
            )
