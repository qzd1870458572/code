from scipy.fft import fft
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from PyEMD import EMD
from vmdpy import VMD
import os

# Fast Fourier Transform
def frequency_feature_extraction(imfs):
    fft_features = []
    for imf in imfs:
        freq_domain = np.abs(fft(imf))
        fft_features.append(freq_domain[:len(freq_domain) // 2])
    return np.array(fft_features)

# k-means clustering
def kmeans_clustering(fft_features, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(fft_features)
    return labels

def combine_imfs_by_cluster(imfs, labels, num_clusters):
    combined_imfs = np.zeros((num_clusters, imfs.shape[1]))
    for i in range(num_clusters):
        combined_imfs[i] = np.sum(imfs[labels == i], axis=0)
    return combined_imfs


file = pd.read_excel("../2010-2019-15-140.xlsx")
data = file['shuju'].values

# Divide the training and test
train_size = int(0.7 * len(data))
data_train = data[:train_size]
data_test = data[train_size:]

# 数据标准化处理
scaler = MinMaxScaler()
train_norm = scaler.fit_transform(data_train.reshape(-1, 1)).flatten()
test_norm = scaler.transform(data_test.reshape(-1, 1)).flatten()

alpha = 1000
tau = 0
K = 10
DC = 0
init = 1
tol = 1e-7

train_IMFs1, _, _ = VMD(train_norm, alpha, tau, K, DC, init, tol)

test_IMFs1, _, _ = VMD(test_norm, alpha, tau, K, DC, init, tol)

fft_features_train = frequency_feature_extraction(train_IMFs1)
fft_features_test = frequency_feature_extraction(test_IMFs1)

num_clusters = 7

train_labels = kmeans_clustering(fft_features_train, num_clusters=num_clusters)
test_labels = kmeans_clustering(fft_features_test, num_clusters=num_clusters)

combined_train_imfs = combine_imfs_by_cluster(train_IMFs1, train_labels, num_clusters=num_clusters)
combined_test_imfs = combine_imfs_by_cluster(test_IMFs1, test_labels, num_clusters=num_clusters)

train_IMFs = combined_train_imfs
test_IMFs = combined_test_imfs

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 20


def create_IMF_sequences(IMFs, seq_length):
    IMF_datasets = []
    for imf in IMFs:
        X_seq, y_seq = create_sequences(imf, seq_length)
        IMF_datasets.append((X_seq, y_seq))
    return IMF_datasets


train_IMF_datasets = create_IMF_sequences(train_IMFs, seq_length)
test_IMF_datasets = create_IMF_sequences(test_IMFs, seq_length)

def IMF_to_tensor(IMF_datasets):
    IMF_tensors = []
    for X_seq, y_seq in IMF_datasets:
        X_tensor = torch.tensor(X_seq.reshape(-1, seq_length, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32)
        IMF_tensors.append((X_tensor, y_tensor))
    return IMF_tensors


train_IMF_tensors = IMF_to_tensor(train_IMF_datasets)
test_IMF_tensors = IMF_to_tensor(test_IMF_datasets)

class CNN_GRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_channels, kernel_size, dropout_prob):
        super(CNN_GRU_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        padding = (kernel_size - 1) // 2

        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(cnn_channels, hidden_size, num_layers=num_layers, batch_first=True)

        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def attention(self, gru_output):
        attention_scores = self.attention_weights(gru_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(attention_weights * gru_output, dim=1)
        return weighted_output

    def forward(self, x):
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = self.relu(self.cnn(x))
        x = x.transpose(1, 2)

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        gru_output, _ = self.gru(x, h_0)

        weighted_output = self.attention(gru_output)

        out = self.dropout(weighted_output)
        out = self.fc(out)
        return out

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
hidden_size = 128
num_layers = 2
cnn_channels = 64
kernel_size = 3
dropout_prob = 0.1
batch_size = 32
epochs = 100

def initialize_gru_model():
    model = CNN_GRU_Attention(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              cnn_channels=cnn_channels,
                              kernel_size=kernel_size,
                              dropout_prob=dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn

def train_and_predict_GRU(train_tensors, test_tensors, epochs):
    all_train_preds = []
    all_test_preds = []

    for (X_train, y_train), (X_test, y_test) in zip(train_tensors, test_tensors):

        model, optimizer, loss_fn = initialize_gru_model()

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train.to(device)).cpu().numpy().flatten()
            y_test_pred = model(X_test.to(device)).cpu().numpy().flatten()

        all_train_preds.append(y_train_pred)
        all_test_preds.append(y_test_pred)

    return np.array(all_train_preds), np.array(all_test_preds)


train_preds, test_preds = train_and_predict_GRU(train_IMF_tensors, test_IMF_tensors, epochs)

train_total_pred = np.sum(train_preds, axis=0)
test_total_pred = np.sum(test_preds, axis=0)

train_total_pred_rescaled = scaler.inverse_transform(train_total_pred.reshape(-1, 1)).flatten()
test_total_pred_rescaled = scaler.inverse_transform(test_total_pred.reshape(-1, 1)).flatten()

y_train_true_rescaled = scaler.inverse_transform(train_norm[seq_length:].reshape(-1, 1)).flatten()
y_test_true_rescaled = scaler.inverse_transform(test_norm[seq_length:].reshape(-1, 1)).flatten()

def plot_IMFs(IMFs, title='IMF Components'):
    num_IMFs = len(IMFs)
    num_imfs = IMFs.shape[0]
    plt.figure(figsize=(10, num_imfs * 2))
    for i in range(num_IMFs):
        plt.subplot(num_IMFs, 1, i + 1)
        plt.plot(IMFs[i])
        plt.title(f'IMF {i+1}')
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

plot_IMFs(train_IMFs, title="Train Data IMFs")
plot_IMFs(test_IMFs, title="Test Data IMFs")

# 评估函数
def evaluate_model(true_values, predicted_values, label='Data'):
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="Original value")
    plt.plot(predicted_values, label="Predictive value", linestyle='--')
    plt.title(f'{label} Data Prediction vs Actual')
    plt.legend()
    plt.show()

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    non_zero_indices = true_values != 0
    mape = np.mean(np.abs(
        (true_values[non_zero_indices] - predicted_values[non_zero_indices]) / true_values[non_zero_indices])) * 100

    print(f'{label} MAE: {mae}')
    print(f'{label} MSE: {mse}')
    print(f'{label} RMSE: {rmse}')
    print(f'{label} R²: {r2}')
    print(f'{label} MAPE: {mape}%')




