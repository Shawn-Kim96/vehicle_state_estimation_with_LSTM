"""
Customize utils.data.Dataset for training
return
    x = [x1, x2, ... , x100]
    y = x101
"""
from torch.utils.data import Dataset


class SinDataset(Dataset):
    def __init__(self, total_data, data_type):
        """

        :param total_data: [[x_list, y], [x_list, y], ...]
        :param data_type:
        :param days_for_forecast:
        """
        self.data_total_num = len(total_data)
        self.data_input_sequence_length = len(total_data[0][0])
        self.data_type = data_type

        # train_last, valid_last : last index number for train, valid (abs value)
        train_data_idx = self.data_total_num*8//10
        valid_data_idx = self.data_total_num*9//10
        if self.data_type == 'train':
            data = total_data[:train_data_idx]
        elif self.data_type == 'valid':
            data = total_data[train_data_idx: valid_data_idx]
        elif self.data_type == 'test':
            data = total_data[valid_data_idx:]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
