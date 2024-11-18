import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import prepare_time_series_data, TimeSeriesDataset, TSModel

def forecast(args):
    """

    Args:
        args:

    Returns:

    """

    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'

    df = pd.read_csv(args.data, low_memory=False)
    data = np.array(df[args.train_column])
    all_dict = prepare_time_series_data(data=data, sequence_length=args.rolling_size)

    temp = all_dict['scaler'].transform(np.array(data[-args.rolling_size:]).reshape(-1,1))
    first_data_to_forecast_on = torch.FloatTensor(temp).unsqueeze(0).permute(0, 2, 1)

    model_args = (
        1,  # num_features; for now only supports univariate time series, so 1 by default
        64,  # conv_out_features
        3,  # conv_kernel_size
        'same',  # conv_padding
        64,  # lstm_hidden_size
        2,  # lstm_layers
        [30, 10, 1],  # linear_neurons
        True,  # lstm_batch_first
    )
    model = TSModel(*model_args)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model = model.to(device)
    model.eval()

    forecasts = []
    ser = first_data_to_forecast_on.to(device)
    print(f'\nStarting forecasting with {args}')
    with torch.no_grad():
        for i in range(args.forecast):
            pred = model(ser).squeeze(0)
            forecasts.append(pred.detach().cpu().numpy().item())
            ser = torch.cat((ser[:, :, 1:], pred.unsqueeze(0).unsqueeze(0)), dim=2)

    forecasts = all_dict['scaler'].inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten().tolist()
    df = pd.DataFrame(forecasts, columns = ['Forecasts'])
    df.to_csv(args.forecast_save, index=False)

    print(f'\nForecasting complete, and saved in {args.forecast_save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forecasting with trained model')
    parser.add_argument('--data', type=str, help='path to time series csv train file')
    parser.add_argument('--train-column', type=str, help='column name of time series data in csv train file')
    parser.add_argument('--rolling-size', type=int, help='rolling size used to train model')
    parser.add_argument('--model', type=str, help='path to trained model file')
    parser.add_argument('--forecast', type=int, help='number of forecast period')
    parser.add_argument('--forecast-save', type=str, default='data/forecasts.csv', help='filepath to save forecasted data as csv')
    parser.add_argument('--device', type=str, default='cuda', help='whether to train to cpu or gpu, default "cuda", change to "cpu" otherwise')

    forecast(parser.parse_args())