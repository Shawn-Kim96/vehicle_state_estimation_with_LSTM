import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm


class TrainValidEvaluate:
    def __init__(self, model, train_loader, valid_loader, test_loader, n_epochs, optimizer, criterion, device, path, logging, writer):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.path = path
        self.logging = logging
        self.writer = writer
        self.avg_train_loss, self.avg_train_mape = [], []
        self.avg_valid_loss, self.avg_valid_mape = [], []
        self.test_prediction_result, self.test_target_result = [], []
        self.avg_test_mape, self.avg_test_loss = np.inf, np.inf

    def sin_evaluate(self, x, y):
        output = self.model(x.to(self.device))
        predicted = output.squeeze().detach()
        mape = mean_absolute_percentage_error(y, predicted.cpu())
        loss = self.criterion(output.squeeze().cpu().float(), y.float())
        return output, predicted, mape, loss

    def sin_train(self):
        best_loss, best_mape = np.inf, np.inf
        # patience = 0

        for epoch in range(1, self.n_epochs + 1):
            train_rmse_losses, valid_rmse_losses = [], []
            train_mape, valid_mape = [], []

            # TRAIN
            self.model.train()
            for i, (x, y) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                output, predicted, mape, loss = self.sin_evaluate(x, y)
                train_mape.append(mape)
                loss.backward()
                self.optimizer.step()
                train_rmse_losses.append(loss.item() ** 0.5)
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()

            # VALIDATION
            self.model.eval()
            with torch.no_grad():
                for x, y in self.valid_loader:
                    output, predicted, mape, loss = self.sin_evaluate(x, y)
                    valid_mape.append(mape)
                    valid_rmse_losses.append(loss ** 0.5)

            train_rmse_losses = np.average(train_rmse_losses)
            valid_rmse_losses = np.average(valid_rmse_losses)
            train_mape = np.average(train_mape)
            valid_mape = np.average(valid_mape)

            self.avg_train_loss.append(train_rmse_losses)
            self.avg_train_mape.append(train_mape)
            self.avg_valid_loss.append(valid_rmse_losses)
            self.avg_valid_mape.append(valid_mape)

            self.writer.add_scalar("ACC_MAPE/train", train_mape, epoch)
            self.writer.add_scalar("LOSS_RMSE/train", train_rmse_losses, epoch)
            self.writer.add_scalar("ACC_MAPE/valid", valid_mape, epoch)
            self.writer.add_scalar("LOSS_RMSE/valid", valid_rmse_losses, epoch)

            print_msg = (f'train_loss: {train_rmse_losses:.4f} train_mape: {train_mape:.4f} || ' +
                         f'valid_loss: {valid_rmse_losses:.4f} valid_mape: {valid_mape:.4f}')
            self.logging.info(print_msg)

            if epoch % 5 == 0:
                if best_mape > valid_mape:
                    best_mape = valid_mape
                    torch.save(self.model.state_dict(), self.path)
                    self.logging.info('Saving best performance model')

        self.model.load_state_dict(torch.load(self.path))

    def sin_test(self):
        test_rmse_losses, test_mape = 0, 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                output, predicted, mape, loss = self.sin_evaluate(x, y)
                self.avg_test_mape += mape
                self.avg_test_loss += (loss ** 0.5)

                for y, p in zip(y, predicted):
                    self.test_prediction_result.append(p)
                    self.test_target_result.append(y)
        self.avg_test_mape /= len(self.test_loader)
        self.avg_test_loss /= len(self.test_loader)

        return self.avg_test_loss, self.avg_test_mape, self.test_prediction_result, self.test_target_result
