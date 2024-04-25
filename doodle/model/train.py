""" Train network here. Persistence to store weights. """

##### Libraries

import numpy as np
import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from model import DoodleANN

# =============================================================================
# Directory Path
# =============================================================================

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH))
CSV_PATH = os.path.join(ROOT_PATH, 'data', 'csv')
ENCODER_PATH = os.path.join(ROOT_PATH, 'data', 'joblib')
TRAINED_NETWORK_PATH = os.path.join(ROOT_PATH, 'doodle', 'trained_network')

# =============================================================================
# Train Network
# =============================================================================

## Get data ready
print("Getting data ready to train...")
df = pd.read_csv(os.path.join(CSV_PATH, 'doodle_dataframe.csv'))
pixel_columns = [f'pixel{i}' for i in range(28*28)]
X = df[pixel_columns].values
y = df['label'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label Encoder to change string categorical labels to numbers
encoder = joblib.load(os.path.join(ROOT_PATH, 'doodle_category_encoder.joblib'))
y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

## Train model
print("Training...")
model = DoodleANN()
# print("Loading weights...")
# model.load_state_dict(torch.load('/content/doodle_ann_weights.pth'))
model.train()  # set model to training mode

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)      # ensure long type for CrossEntropyLoss

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Training Loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')

    if epoch % 10 == 0:
      # Save weights
      print("Persistence: Saving model weights...")
      torch.save(model.state_dict(), os.path.join(TRAINED_NETWORK_PATH, f'doodle_ann_weights_{str(epochs)}.pth'))

# Save weights
print("Finished training. Saving model weights!")
torch.save(model.state_dict(), os.path.join(TRAINED_NETWORK_PATH, f'doodle_ann_weights_{str(epochs)}.pth'))