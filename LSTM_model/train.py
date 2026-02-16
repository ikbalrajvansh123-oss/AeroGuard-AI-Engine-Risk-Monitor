import joblib
from tqdm import tqdm
import torch.nn as nn
import torch
from data_loader import load_data
from classify_risk import classify_risk
from scaling_data import scaling_data
from create_sequence import create_sequences
from split_data import split_data
from model import LSTMClassifier


def train(model, train_loader, criterion, optimizer, epochs,device):
  for epoch in range(epochs):
      model.train()
      total_loss = 0

      for xb, yb in tqdm(train_loader):
          xb, yb = xb.to(device), yb.to(device)
          optimizer.zero_grad()

          outputs = model(xb)
          loss = criterion(outputs, yb)

          loss.backward()
          optimizer.step()

          total_loss += loss.item()

      print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")




from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test, device):
  model.eval()
  with torch.no_grad():
      X_test_gpu = X_test.to(device).float()
      outputs = model(X_test_gpu)

      _, predicted = torch.max(outputs, 1)

  predicted = predicted.cpu().numpy()
  y_test_numpy = y_test.cpu().numpy()
  accuracy = accuracy_score(y_test_numpy, predicted)
  print(f"Test Accuracy: {accuracy:.4f}")
  print(classification_report(y_test_numpy, predicted))
  return predicted, y_test_numpy, outputs

def main():
  train_df = load_data()
  train_df['risk_label'] = train_df['RUL'].apply(classify_risk)
  train_df,scaler,feature_cols = scaling_data(train_df)

  sequence_length = 30
  X, y = create_sequences(train_df, feature_cols,sequence_length)

  batch_size = 64
  train_loader,X_test,y_test = split_data(X,y,batch_size)

  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = LSTMClassifier(
      input_size=len(feature_cols),
      hidden_size=64,
      num_layers=2,
      num_classes=3
  ).to(device)
  class_weights = torch.tensor([1.0, 1.2, 1.0]).to(device)

  criterion = nn.CrossEntropyLoss(weight=class_weights)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  epochs = 15
  # batch_size = 64 # Already defined
  train(model, train_loader, criterion, optimizer, epochs,device)
  torch.save(model.state_dict(), "models/risk_lstm_model.pth")
  joblib.dump(scaler, "model/scaler.pkl")

  metadata = {
          "feature_cols": feature_cols,
          "sequence_length": sequence_length
      }

  joblib.dump(metadata, "model/metadata.pkl")
  predicted, y_test_numpy, outputs = evaluate(model, X_test, y_test, device)
  return predicted, y_test_numpy, outputs

if __name__ == "__main__":
  predicted_global, y_test_numpy_global, outputs_global = main()
