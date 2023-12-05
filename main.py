# -------------IMPORTS------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from GRUModel import GRUModel
from LSTMModel import LSTMModel, BidirectionalLSTM

# ----------IMPORT THE DATASET--------
csv_path = 'Cardano.csv'
dataset_train = pd.read_csv(csv_path)
training_set = dataset_train.iloc[:, 1:2].values

print(dataset_train.head())

# -----------EXTRACT THE OPENING PRICES AND SCALE THEM------
# Extract the 'Open' prices
open_prices = dataset_train['Open'].values

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_open_prices = scaler.fit_transform(open_prices.reshape(-1, 1))


# -------CREATING SEQUENCES FOR LSTM WITH SEQ_LEN=3-------
# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


# Choose the sequence length (number of time steps to consider for each input)
sequence_length = 3

# Create sequences
sequences = create_sequences(scaled_open_prices, sequence_length)

# Display the first few sequences
print("\nScaled Open Prices:")
print(scaled_open_prices)
print("\nSequences:")
print(sequences)

# Define the target variable (next day's 'Open' price)
target = scaled_open_prices[sequence_length:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, target, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

# Instantiate the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 50
num_layers = 1
model = BidirectionalLSTM(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=4e-4)

# Convert data to DataLoader
batch_size = 32
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 450
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)

        # Compute the loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)

# Calculate the test loss
test_loss = criterion(test_outputs, y_test_tensor)
print(f'Test Loss: {test_loss.item():.10f}')

# Convert predictions and true values to NumPy arrays
test_outputs = test_outputs.numpy()
y_test_numpy = y_test_tensor.numpy()

# Calculate and print the Mean Squared Error on the test set
mse = mean_squared_error(y_test_numpy, test_outputs)
print(f'Mean Squared Error on Test Set: {mse:.10f}')

rmse_lstm = np.sqrt(mse)
print(f'Root Mean Squared Error on Test Set (LSTM): {rmse_lstm:.10f}')


# Calculate and print the Mean Absolute Error on the test set
mae = np.mean(np.abs(y_test_numpy - test_outputs))
print(f'Mean Absolute Error on Test Set: {mae:.10f}')

# Convert the test outputs to a NumPy array
test_outputs = test_outputs.squeeze()

# Denormalize the scaled values to get the original prices
predicted_prices = scaler.inverse_transform(test_outputs.reshape(-1, 1))
true_prices = scaler.inverse_transform(y_test_numpy.reshape(-1, 1))

# Plot the predictions and true prices with dates
plt.figure(figsize=(12, 6))

# Extract dates for the test set
test_dates = dataset_train.iloc[-len(true_prices):]['Date'].values

# Convert 'Date' column to datetime format
dataset_train['Date'] = pd.to_datetime(dataset_train['Date'])

# Plot the predicted prices
plt.plot(dataset_train['Date'].iloc[-len(true_prices):], predicted_prices, label='Predicted Prices', marker='o',
         color='blue')

# Plot the true prices
plt.plot(dataset_train['Date'].iloc[-len(true_prices):], true_prices, label='True Prices', marker='o',
         color='red')

plt.title('LSTM Predictions vs True Prices')
plt.xlabel('Date')
plt.ylabel('Open Prices')
plt.legend()

# Set x-axis ticks to display every 1 months
months = MonthLocator(interval=1)
months_fmt = DateFormatter("%Y-%m-%d")  # Adjust the date format as needed
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(months_fmt)

# Add a legend specifying which color is which price
plt.legend(['GRU Predicted Prices (Blue)', 'True Prices (Red)'])
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

# Use the last sequence from the original data as a starting point for prediction
last_sequence = scaled_open_prices[-sequence_length:]

# Convert the last sequence to a PyTorch tensor
last_sequence_tensor = torch.Tensor(last_sequence.reshape(1, sequence_length, 1))

# Make a prediction for the next day
with torch.no_grad():
    model.eval()
    next_day_prediction = model(last_sequence_tensor)

# Denormalize the predicted value to get the original price
next_day_price = scaler.inverse_transform(next_day_prediction.numpy().reshape(-1, 1))

print("Predicted Price for the Next Day:", next_day_price[0, 0])


# Instantiate the GRU model
gru_model = GRUModel(input_size, hidden_size, num_layers)

# Define loss function and optimizer for GRU model
gru_criterion = nn.MSELoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=1e-3)

# Convert data to DataLoader for GRU model
gru_train_data = TensorDataset(X_train_tensor, y_train_tensor)
gru_train_loader = DataLoader(gru_train_data, batch_size=batch_size, shuffle=True)

# Train the GRU model
for epoch in range(num_epochs):
    for batch_x, batch_y in gru_train_loader:
        # Forward pass
        gru_outputs = gru_model(batch_x)

        # Compute the loss
        gru_loss = gru_criterion(gru_outputs, batch_y)

        # Backward pass and optimization
        gru_optimizer.zero_grad()
        gru_loss.backward()
        gru_optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], GRU Loss: {gru_loss.item():.10f}')

# Evaluate the GRU model on the test set
with torch.no_grad():
    gru_model.eval()
    gru_test_outputs = gru_model(X_test_tensor)

# Calculate the test loss for GRU model
gru_test_loss = gru_criterion(gru_test_outputs, y_test_tensor)
print(f'GRU Test Loss: {gru_test_loss.item():.10f}')

# Convert predictions and true values to NumPy arrays for GRU model
gru_test_outputs = gru_test_outputs.numpy()
gru_y_test_numpy = y_test_tensor.numpy()

# Calculate and print the Mean Squared Error on the test set for GRU model
gru_mse = mean_squared_error(gru_y_test_numpy, gru_test_outputs)
print(f'Mean Squared Error on Test Set (GRU): {gru_mse:.10f}')

# Calculate RMSE for GRU
rmse_gru = np.sqrt(gru_mse)
print(f'Root Mean Squared Error on Test Set (GRU): {rmse_gru:.10f}')

# Calculate and print the Mean Absolute Error on the test set for GRU model
gru_mae = np.mean(np.abs(gru_y_test_numpy - gru_test_outputs))
print(f'Mean Absolute Error on Test Set (GRU): {gru_mae:.10f}')

# Convert the GRU test outputs to a NumPy array
gru_test_outputs = gru_test_outputs.squeeze()

# Denormalize the scaled values to get the original prices for GRU model
gru_predicted_prices = scaler.inverse_transform(gru_test_outputs.reshape(-1, 1))
gru_true_prices = scaler.inverse_transform(gru_y_test_numpy.reshape(-1, 1))

# Plot the GRU predictions and true prices with dates
plt.figure(figsize=(12, 6))

# Extract dates for the test set
test_dates = dataset_train.iloc[-len(gru_true_prices):]['Date'].values

# Convert 'Date' column to datetime format
dataset_train['Date'] = pd.to_datetime(dataset_train['Date'])

# Plot the GRU predicted prices
plt.plot(dataset_train['Date'].iloc[-len(gru_true_prices):], gru_predicted_prices, label='GRU Predicted Prices', marker='o', color='blue')

# Plot the true prices for GRU
plt.plot(dataset_train['Date'].iloc[-len(gru_true_prices):], gru_true_prices, label='True Prices', marker='o',
         color='red')

plt.title('GRU Predictions vs True Prices')
plt.xlabel('Date')
plt.ylabel('Open Prices')
plt.legend()

# Set x-axis ticks to display every 1 month for GRU model
months = MonthLocator(interval=1)
months_fmt = DateFormatter("%Y-%m-%d")  # Adjust the date format as needed
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(months_fmt)

# Add a legend specifying which color is which price
plt.legend(['GRU Predicted Prices (Blue)', 'True Prices (Red)'])

plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

# Use the last sequence from the original data as a starting point for prediction with GRU model
last_sequence_gru = scaled_open_prices[-sequence_length:]

# Convert the last sequence to a PyTorch tensor for GRU model
last_sequence_tensor_gru = torch.Tensor(last_sequence_gru.reshape(1, sequence_length, 1))

# Make a prediction for the next day with GRU model
with torch.no_grad():
    gru_model.eval()
    next_day_prediction_gru = gru_model(last_sequence_tensor_gru)

# Denormalize the predicted value to get the original price with GRU model
next_day_price_gru = scaler.inverse_transform(next_day_prediction_gru.numpy().reshape(-1, 1))

print("Predicted Price for the Next Day (GRU):", next_day_price_gru[0, 0])
