### Argument Values
- model_name=F_uniform_LSTM
- data_dir=data.csv
- device=cpu
- input_sequence_length=40
- hidden_dim1=100
- loss=L2
- batch_size=32
- num_layers=1
- weight_decay=1e-06
- learning_rate=0.001
- n_epochs=500
- logging_level=info
- predict_column=state

### Model Architecture
```FLSTM(
  (lstm): LSTM(6, 100, batch_first=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (relu): ReLU()
  (bn0): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc0): Linear(in_features=100, out_features=50, bias=True)
  (fc1): Linear(in_features=50, out_features=3, bias=True)
)```

### Parameter Number
- 48503
