import math
import numpy as np
import random


def generate_sin_data(d_num, data_interval):
    t_list = [i/data_interval for i in range(d_num)]
    sin_list = [math.sin(t) for t in t_list]

    input_dim = int(10)
    dataset_num = d_num - input_dim

    xy_list = [(np.array([[x] for x in sin_list[i:i + input_dim]]), np.array(sin_list[i + input_dim]))
               for i in range(dataset_num)]
    random.shuffle(xy_list)
    return xy_list
