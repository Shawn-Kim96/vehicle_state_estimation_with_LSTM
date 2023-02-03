import matplotlib.pyplot as plt


class VisualizeUtils:
    def __init__(self, predict_value, true_value):
        self.predict_value = predict_value
        self.true_value = true_value

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
            x = [j for j in range(len(pred_[0]))]
            plt.plot(x, pred_[0], color='blue', label='input')
            plt.scatter(len(pred_[0]), pred_[1], color='red', label='predict')
            plt.scatter(len(pred_[0]), true_[1], color='green', label='true')
            plt.legend()
            plt.ylim([-1, 1])
            plt.show()
            if i > 5:
                break
