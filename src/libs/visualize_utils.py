import matplotlib.pyplot as plt
import numpy as np


class VisualizeUtils:
    def __init__(self, predict_value, true_value, data_interval):
        self.predict_value = predict_value
        self.true_value = true_value
        self.data_step =  1/data_interval

    def visualize_1d_data(self, plot_title):
        plt.figure()
        plt.plot(range(len(self.predict_value)), self.predict_value, label='predict')
        plt.plot(range(len(self.true_value)), self.true_value, label='true')
        plt.title(plot_title)
        plt.legend()
        plt.show()

    def visualize_sin_data(self, plot_title):
        for i, (pred_, true_) in enumerate(zip(self.predict_value, self.true_value)):
            plt.figure()
            plt.title(f"{plot_title}_data{i:02d}")
            x = np.arcsin(pred_[0]).squeeze()
            x_next = x[-1] + self.data_step
            x_total = np.append(x, x_next)
            plt.plot(x, pred_[0], color='blue', label='input')
            plt.scatter(x_next, pred_[1], color='red', label='predict')
            plt.scatter(x_next, true_[1], color='green', label='true')

            # guideline for sin and cos
            guideline_x = np.arange(min(x_total), max(x_total), 1e-4)
            plt.plot(guideline_x, np.sin(guideline_x), '--', label='sin', color='blue', alpha=0.5)
            plt.plot(guideline_x, np.cos(guideline_x), '--', label='cos', color='red', alpha=0.5)
            plt.legend()
            plt.ylim([-1.2, 1.2])
            plt.show()
            if i > 5:
                break
