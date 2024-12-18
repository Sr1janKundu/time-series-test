# Train and forecast cnn-lstm model on time series data

Clone, train and get forecast on any time series csv file. The model is:

```
TSModel(
  (conv1d): Conv1d(1, 64, kernel_size=(3,), stride=(1,), padding=same)
  (lstm): LSTM(64, 64, num_layers=2, batch_first=True)
  (fcs): Sequential(
    (0): Linear(in_features=64, out_features=30, bias=True)
    (1): ReLU()
    (2): Linear(in_features=30, out_features=10, bias=True)
    (3): ReLU()
    (4): Linear(in_features=10, out_features=1, bias=True)
  )
)
```


## Usage

Run the `train.py` file with required arguments.

```bash
python train.py --data <location-to-train-csv> --train-column <column-name> --rolling-size <number of time points to be trained in one go> --epoches <num_epoches> --batch-size <batch_size> --lr <learnig_rate>
```

The model will be saved in the `model/` directory. Now to get forecast, run the `forecast.py` file with required arguments.

```bash
python forecast.py --data <location-to-train-csv> --train-column <column-name> --rolling-size <rolling size used to train model> --model <path to trained model file> --forecast <number of timepoints to forecast> --forecast-save <filepath to save forecasted data as csv>
```

The forecasted data will be saved as csv at given location.


## Requirements

This is written with python version 3.10.14, with following libraries:

- numpy==2.0.2
- pandas==2.2.3
- scikit-learn==1.5.2
- pytorch==2.5.1

## To-Do

- [ ] Train the model on whole data file, not on only train split.
- [ ] Add appropriate datetime with forecasted data.