# vehicle_state_estimation_with_LSTM
Vehicle state estimation with LSTM, using sensor data
---
Directory Structure

```
├── README.md
│
├── data               <- Uploaded to Google Drive or Storage. Not uploaded to Gitlab
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── models             <- Trained and serialized models (rule-base, ML, DL models included)
│
├── results            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │   
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models         <- Scripts for main model predictions. Models are imported from Project/model
    │   ├── train_model.py
    │   └── predict_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```