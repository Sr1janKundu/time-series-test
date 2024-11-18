import matplotlib.pyplot as plt
from IPython import embed
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import prepare_time_series_data, TimeSeriesDataset, TSModel


def main(args):
    """

    Args:
        args:

    Returns:

    """

    # data, train_window, batch_size, model, model_args, num_epoch, lr = 1e-4, device='cpu'):
    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'

    df = pd.read_csv(args.data, low_memory=False)
    data = np.array(df[args.train_column])
    all_dict = prepare_time_series_data(data=data, sequence_length=args.rolling_size)

    train_dataset = TimeSeriesDataset(all_dict['train_sequences'], all_dict['train_targets'])
    val_dataset = TimeSeriesDataset(all_dict['test_sequences'], all_dict['test_targets'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model_args = (
        1,  # num_features; for now only supports univariate time series, so 1 by default
        64, # conv_out_features
        3,  # conv_kernel_size
        'same', # conv_padding
        64, # lstm_hidden_size
        2,  # lstm_layers
        [30, 10, 1],    # linear_neurons
        True,   # lstm_batch_first
    )
    model = TSModel(*model_args).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_loss = 0.0
    
    print(f"\nStarting training with {args}.")
    for epoch in range(args.epoches):
        print(f'\nEpoch: {epoch+1}')
        for idx, (batch_data, batch_targets) in tqdm(enumerate(train_loader)):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epoches}], Loss: {total_loss/len(train_loader):.4f}')

    model_file = args.model + os.path.basename(args.data).split('.csv')[0] + '.pth'
    torch.save(model.state_dict(), model_file)
    print(f'\n\n--Model saved at {model_file} --')
    # validation
    # Make predictions
    model.eval()

    predictions = []
    actuals = []
    print("\n\nValidation: ")
    with torch.no_grad():
        for idx, (sequences, targets) in tqdm(enumerate(val_loader)):
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            print(f"Loss: {criterion(outputs, targets).item()}")
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    #
    # Inverse transform predictions and actuals
    predictions = all_dict['scaler'].inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = all_dict['scaler'].inverse_transform(np.array(actuals).reshape(-1, 1))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Time Series Forecasting Results')
    plt.savefig(f"val_graphs\\{os.path.basename(args.data).split('.csv')[0]}_validation.png")
    plt.show()

    # return predictions, actuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time series modelling and forecasting with Conv1d-LSTM (rolling window)")
    parser.add_argument('--data', type=str, help='path to time series csv train file')
    parser.add_argument('--train-column', type=str, help='column name of time series data in csv train file')
    parser.add_argument('--rolling-size', type=int, help='number of data points to be trained at one go')
    parser.add_argument('--model', type=str, default='model/', help='directory to store model file')
    parser.add_argument('--epoches', type=int, default=100, help='number of epoches to train model')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of the model (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda', help='whether to train to cpu or gpu, default "cuda", change to "cpu" otherwise')
    main(parser.parse_args())