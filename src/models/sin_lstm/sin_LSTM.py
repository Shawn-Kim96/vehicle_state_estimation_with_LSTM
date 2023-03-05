import torch.nn as nn


class SinLSTM(nn.Module):
    def __init__(
        self,
        input_dim=100,
        lstm_hidden_dim=128,
        output_dim=1,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.hidden_dim = lstm_hidden_dim * num_layers

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.bn0 = nn.BatchNorm1d(self.hidden_dim // 2)

        self.fc0 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc1 = nn.Linear(self.hidden_dim // 2, output_dim)

    def forward(self, seq_x):
        _, (h_t, c_t) = self.lstm(
            seq_x.float()
        )  # h_t dim = [num_layers, batch_size, LSTM_output_dim]
        output = h_t.permute(
            1, 0, 2
        )  # output dim = [batch_size, num_layers, LSTM_output_dim]
        output = self.flatten(
            output
        )  # output dim = [batch_size, num_layers*LSTM_output_dim]
        output = self.relu(
            self.bn0(self.fc0(output))
        )  # output = [batch_size, fc0_output_dim]
        output = self.fc1(output)  # output = [batch_size, fc1_output_dim]

        return output
