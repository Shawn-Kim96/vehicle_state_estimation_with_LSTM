import math
import random

import numpy as np


def generate_sin_data(d_num, data_interval, data_sequence):
    t_list = [i / data_interval for i in range(d_num)]
    sin_list = [math.sin(t) for t in t_list]

    input_dim = data_sequence
    dataset_num = d_num - input_dim

    txy_list = [
        (
            np.array([[t] for t in t_list[i : i + input_dim + 1]]),
            np.array([[x] for x in sin_list[i : i + input_dim]]),
            np.array(sin_list[i + input_dim]),
        )
        for i in range(dataset_num)
    ]
    random.shuffle(txy_list)
    return txy_list


def generate_cos_to_sin_data(d_num, data_interval, data_sequence):
    t_list = [i / data_interval for i in range(d_num)]
    sin_list = [math.sin(t) for t in t_list]
    cos_list = [math.cos(t) for t in t_list]

    input_dim = data_sequence
    dataset_num = d_num - input_dim

    txy_list = [
        (
            np.array([[t] for t in t_list[i : i + input_dim + 1]]),
            np.array([[x] for x in sin_list[i : i + input_dim]]),
            np.array(cos_list[i + input_dim]),
        )
        for i in range(dataset_num)
    ]
    random.shuffle(txy_list)
    return txy_list
