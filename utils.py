import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def prepare_time_series_data(data, sequence_length,  train_split=0.8
                             ):
    """
    Prepare time series data for training and testing
    Args:
        data: Input time series data, if passing as pandas series, pass series.values
        sequence_length (int): Length of input sequences
        train_split (float): Proportion of data to use for training

    Returns:
        dict: Dictionary containing training and testing datasets, as well as scaler
    """
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))  # reshaping required for scaling

    # Create sequences
    sequences = []
    targets = []

    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        targets.append(scaled_data[i + sequence_length])

    sequences = np.array(sequences)
    targets = np.array(targets)

    # sequences = torch.FloatTensor(np.array(sequences))
    # targets = torch.FloatTensor(np.array(targets))

    # Split into training and testing sets
    # train_size = int(len(sequences) * train_split)
    # modification
    train_size = int((len(data)*train_split // sequence_length) * sequence_length)

    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]

    # Convert to PyTorch tensors
    train_sequences = torch.FloatTensor(train_sequences)
    train_targets = torch.FloatTensor(train_targets)
    test_sequences = torch.FloatTensor(test_sequences)
    test_targets = torch.FloatTensor(test_targets)

    return {
        'train_sequences': torch.permute(train_sequences, (0, 2, 1)),
        'train_targets': train_targets,
        'test_sequences': torch.permute(test_sequences, (0, 2, 1)),
        'test_targets': test_targets,
        # 'train_sequences': torch.permute(sequences, (0, 2, 1)),
        # 'train_targets': targets,
        'scaler': scaler
    }


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        """

        Args:
            sequences (torch.tensor):
            targets (torch.tensor):
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TSModel(nn.Module):
    def __init__(self, num_features, conv_out_features, conv_kernel_size, conv_padding, lstm_hidden_size, lstm_layers,
                 linear_neurons, lstm_batch_first=True):
        """

        Args:
            num_features:
            conv_out_features:
            conv_kernel_size:
            conv_padding:
            lstm_hidden_size:
            lstm_layers:
            linear_neurons:
            lstm_batch_first:
        """
        super(TSModel, self).__init__()
        self.num_features = num_features
        self.conv_out_features = conv_out_features
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.fc1, self.fc2, self.fc3 = linear_neurons
        self.lstm_batch_first = lstm_batch_first

        self.conv1d = nn.Conv1d(in_channels=self.num_features,
                                out_channels=self.conv_out_features,
                                kernel_size=self.conv_kernel_size,
                                padding=self.conv_padding)
        self.lstm = nn.LSTM(input_size=self.conv_out_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=self.lstm_batch_first)
        self.fcs = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc1),
            nn.ReLU(),
            nn.Linear(self.fc1, self.fc2),
            nn.ReLU(),
            nn.Linear(self.fc2, self.fc3)
        )

    def forward(self, x):
        o1 = self.conv1d(x)
        # o1 is of size [batch, self.conv_out_features, x.shape[2]]

        o1_res = torch.permute(o1, (0, 2, 1))
        o2, (_, _) = self.lstm(o1_res)
        # o2 is of size  [batch, x.shape[2], self.lstm_hidden_size]

        # take last time stamp of o2
        o2_final = o2[:, -1, :]
        # o2_final size [batch, self.lstm_hidden_size]

        o3 = self.fcs(o2_final)

        return o3