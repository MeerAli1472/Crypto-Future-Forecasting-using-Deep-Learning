import numpy as np
import torch
import torch.nn as nn



class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Recurrent layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Added layers (MLP head)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.rnn(x)

        # Take last timestep
        out = out[:, -1, :]   # (batch, hidden_size)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc_out(out)
        return out
        
#================================================================================================================

def train_rnn(num_epochs,model, train_loader, device, optimizer, criterion):
    EPOCHS = num_epochs

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"RNN Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}")

#=====================================================================================================

def get_predictions_rnn(model, loader, scaler_y, device):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)

            preds.append(pred.cpu().numpy())
            actuals.append(y_batch.numpy())

    preds = scaler_y.inverse_transform(np.vstack(preds))
    actuals = scaler_y.inverse_transform(np.vstack(actuals))

    return actuals, preds

#====================================================================================================



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Recurrent layer (LSTM instead of RNN)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # MLP head (unchanged)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.lstm(x)   # out: (batch, timesteps, hidden)

        # Take last timestep
        out = out[:, -1, :]    # (batch, hidden_size)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc_out(out)
        return out
        
        
#=====================================================================================================================================
def train_lstm(num_epochs,model,train_loader, device, optimizer, criterion):
    
    EPOCHS = num_epochs

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"LSTM Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}")
        
#=====================================================================================================================================

def get_predictions_lstm(model, loader, scaler_y, device):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)

            preds.append(pred.cpu().numpy())
            actuals.append(y_batch.numpy())

    preds = scaler_y.inverse_transform(np.vstack(preds))
    actuals = scaler_y.inverse_transform(np.vstack(actuals))

    return actuals, preds
