import matplotlib.pyplot as plt
import numpy as np


class VisualizeUtils:
    def __init__(self, predict_value, true_value, data_interval):
        self.predict_value = predict_value
        self.true_value = true_value
        self.data_step = 1/data_interval

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
            t_list = pred_[0]
            x_list = pred_[1]
            predict_value, true_value = pred_[-1], true_[-1]
            plt.scatter(t_list[:-1], x_list, color='blue', label='input')
            plt.scatter(t_list[-1], predict_value, color='red', label='predict')
            plt.scatter(t_list[-1], true_value, color='green', label='true')

            # guideline for sin and cos
            guideline_t = np.arange(min(t_list), max(t_list), 1e-4)
            plt.axhline(y=0, alpha=0.8, color='black')
            plt.plot(guideline_t, np.sin(guideline_t), '--', label='sin', color='blue', alpha=0.5)
            plt.legend()
            plt.ylim([-1.2, 1.2])
            plt.show()
            if i > 5:
                break

    def visualize_sin2cos_data(self, plot_title):
        for i, (pred_, true_) in enumerate(zip(self.predict_value, self.true_value)):
            plt.figure()
            plt.title(f"{plot_title}_data{i:02d}")
            t_list = pred_[0]
            x_list = pred_[1]
            predict_value, true_value = pred_[-1], true_[-1]
            plt.scatter(t_list[:-1], x_list, color='blue', label='input')
            plt.scatter(t_list[-1], predict_value, color='red', label='predict')
            plt.scatter(t_list[-1], true_value, color='green', label='true')

            # guideline for sin and cos
            guideline_t = np.arange(min(t_list), max(t_list), 1e-4)
            plt.axhline(y=0, alpha=0.8, color='black')
            plt.plot(guideline_t, np.sin(guideline_t), '--', label='sin', color='blue', alpha=0.5)
            plt.plot(guideline_t, np.cos(guideline_t), '--', label='cos', color='red', alpha=0.5)
            plt.legend()
            plt.ylim([-1.2, 1.2])
            plt.show()
            if i > 5:
                break

