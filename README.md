# Compressed FL Univariate LSTM Forecasting

> **Historical exploratory prototype.** The current example script takes the first value of each dataset-level `target` series and creates sliding windows across rows rather than consecutive hourly observations. Its reported MAE values therefore should not be treated as validated hourly electricity-load forecasting results. Proper temporal reconstruction and chronological evaluation are needed before drawing performance conclusions. The code is retained for inspection.


Open-source implementation of **compressed federated learning** for **univariate** time-series forecasting using an LSTM on the [Electricity Load Diagrams](https://huggingface.co/datasets/tulipa762/electricity_load_diagrams) dataset.

## How this differs from [Compressed-Federated-Time-Series-Forecasting](https://github.com/emadhammami/Compressed-Federated-Time-Series-Forecasting)

| | Previous repo | This repo |
|---|---|---|
| **Input** | 4 features (`feat_static_cat`, `hour`, `dayofweek`, `month`) | Raw load values only (univariate) |
| **Sequence length** | 10 steps | 24 steps (past 24 h → predict next h) |
| **Local epochs** | 2 | 10 |
| **Model input shape** | `(10, 4)` | `(24, 1)` |

> **Note:** Neither version uses an LLM (Large Language Model). Both use an **LSTM** (Long Short-Term Memory), a classical recurrent neural network for sequential data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python federated_lstm_univariate.py
```

The script will:
1. Download the dataset automatically via Hugging Face `datasets`
2. Train the federated LSTM for 100 rounds across 3 clients
3. Print per-round upload size (KB) and validation MAE
4. Save uplink stats to `federated_lstm_time_series.csv`

## License

MIT
