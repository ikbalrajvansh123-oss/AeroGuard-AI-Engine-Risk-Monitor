import torch
import numpy as np
from LSTM_model.model import LSTMClassifier

# ===== MODEL CONFIG (MATCH TRAINING) =====
INPUT_SIZE = 17
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 3
MODEL_PATH = "models/risk_lstm_model.pth"

# ===== LOAD MODEL =====
device = torch.device("cpu")

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def predict_risk(input_array):

    input_array = np.array(input_array)

    # If single row (manual mode)
    if input_array.shape == (17,):
        input_array = np.tile(input_array, (30, 1))  # (30,17)

    # Add batch dimension
    input_array = np.expand_dims(input_array, axis=0)  # (1,30,17)

    input_tensor = torch.tensor(input_array, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.numpy()[0]
