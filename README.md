# Vehicle Dynamics State Prediction using LSTM Models

This project aims to predict the next state of a vehicle using dynamic data from 6 degrees of freedom (6DOF) and other related vehicle dynamics inputs. 

The project leverages various machine learning models, particularly LSTM (Long Short-Term Memory) networks, to improve the accuracy of state predictions using data such as velocity, acceleration, and position. These models are trained to forecast future states based on past dynamic behavior.


## Project Structure

```plaintext
.
├── README.md                # Project overview
├── data                     # Contains processed datasets for model training
│   └── processed
│       └── 6dof             # 6DOF datasets in CSV and markdown formats
├── dev_utils                # Utilities for handling and processing new datasets
│   └── new_data_file_handler.py
├── env_config               # Configuration settings for the environment
│   ├── __init__.py
│   └── env_config_reader.py
├── models                   # Directory for model architectures and trained models
│   ├── Cos2SinLSTM          # LSTM models predicting cosine from sine
│   ├── F_uniform_LSTM       # LSTM models for state prediction using uniform data
│   ├── LSTM_chirp_signal    # LSTM models trained on chirp signals
│   ├── LSTM_predict_cos_from_sin # Model predicting cosine values from sine inputs
│   ├── LSTM_predict_sum_from_signals # Models for predicting the sum of input signals
├── notebooks                # Jupyter notebooks for model training and experiments
│   └── *.ipynb              # Detailed experiments for each model
├── pyproject.toml           # Project dependencies and configuration
└── src                      # Source code for dataset processing and model training
    ├── data                 # Scripts for generating and handling vehicle dynamics data
    ├── libs                 # Utility libraries for batch processing, data handling, visualization
    └── models               # Code for training LSTM models
```

## Key Components

1. **Data**: The data/processed/6dof folder contains the pre-processed datasets in CSV format, which are used for training the LSTM models. These datasets capture key aspects of vehicle dynamics like velocity, acceleration, and force data.

2. **Models**: This folder houses different LSTM models, each addressing specific prediction tasks, such as predicting vehicle states based on velocity and acceleration data (F_uniform_LSTM) or using sine and cosine signal relationships (Cos2SinLSTM).

3. **Notebooks**: Jupyter notebooks are used to conduct experiments with various model architectures, such as predicting the vehicle's future state using 6DOF data or exploring alternative signal prediction models.

4. **Source Code**: The src directory contains the core code for processing dynamic vehicle data and running the model training pipeline. Key subfolders include:
- `data`: Contains the dynamics data generation code.
- `libs`: Provides utility functions for handling datasets, visualizing results, and interacting with cloud storage.
- `models`: Houses the main training code for the LSTM models, along with specific implementations for different architectures.
- `Development Utilities`: Tools like new_data_file_handler.py are used for managing and processing new datasets.

## Objective
The main goal of this project is to leverage machine learning, particularly LSTM networks, to predict the future state of vehicles based on historical dynamics data. 

By accurately forecasting these states, this system can be used in applications such as autonomous driving, safety simulations, or vehicle dynamics control systems.

## Usage
- `Model Training`: Use the notebooks to load datasets from data/processed/6dof/ and train various LSTM models located in the models/ folder.
- `Experimentation`: Modify the architecture and parameters within the provided notebooks to explore different vehicle state prediction methods.
- `Data Handling`: The dev_utils/ and src/ directories provide scripts for processing new vehicle dynamics data and integrating it into the model training pipeline.
