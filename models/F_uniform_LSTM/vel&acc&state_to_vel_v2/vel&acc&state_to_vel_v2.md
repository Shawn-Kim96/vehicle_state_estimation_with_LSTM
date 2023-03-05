### Argument Values
- model_name=F_uniform_LSTM
- data_dir=data.csv
- device=mps:0
- input_dim=20
- hidden_dim1=100
- loss=L2
- batch_size=16
- num_layers=5
- weight_decay=1e-06
- learning_rate=0.001
- n_epochs=500
- logging_level=info
- predict_column=vel3
- epoch=3000
- input_sequence_length=1

### Model Architecture
```FLSTM(
  (lstm): LSTM(6, 128, batch_first=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (relu): ReLU()
  (bn0): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc0): Linear(in_features=128, out_features=64, bias=True)
  (fc1): Linear(in_features=64, out_features=3, bias=True)
)```

### Parameter Number
- 78211