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
