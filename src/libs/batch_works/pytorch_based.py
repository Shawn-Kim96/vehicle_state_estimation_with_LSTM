import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.trigonometric.trigonometric_dataset import *
from src.libs.dataset_utils import SinDataset
from src.libs.train_valid_test_utils import TrainValidEvaluate
from src.libs.visualize_utils import VisualizeUtils
from src.models.sin_lstm.sin_LSTM import SinLSTM

# from torch.utils.tensorboard import SummaryWriter


np.random.seed(42)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main():
    """
    main function for training, testing model
    :return:
    """
    # Setting log -> log will be saved to log/log.text
    # if not os.path.exists('./log'):
    #     os.makedirs('./log')

    # logging.basicConfig(filename='./log/log.text', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="default_configs.json", type=str)
    parser.add_argument("--data_dir", default="data.csv", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--model_name", default="Cos2SinLSTM", type=str)
    parser.add_argument("--hidden_dim1", default=100, type=int)
    parser.add_argument("--loss", default="L2", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--logging_level", default="info", type=str)
    parser.add_argument("--sin_data_number", default=1000, type=int)
    parser.add_argument("--sin_data_interval", default=20, type=int)
    parser.add_argument("--data_sequence_length", default=10, type=int)
    args = parser.parse_args()

    logging_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger.setLevel(logging_dict[args.logging_level])
    # # Get default value from json
    # with open(args.config_path, 'r') as f:
    #     t_args = argparse.Namespace()
    #     t_args.__dict__.update(json.load(f))
    #     args = parser.parse_args(namespace=t_args)

    save_name = f"{args.model_name}_hd1_{args.hidden_dim1}_nl_{args.num_layers}_lr_{args.learning_rate}"

    save_dir = f"./models/sin_test/{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.info(f"=========MODEL {save_name} TRAIN/TEST START=========")
    # writer = SummaryWriter(log_dir=f'runs/{save_name}')
    logging.info(args)

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.model_name == "SinLSTM":
        total_data = generate_sin_data(
            d_num=args.sin_data_number,
            data_interval=args.sin_data_interval,
            data_sequence=args.data_sequence_length,
        )
    elif args.model_name == "Cos2SinLSTM":
        total_data = generate_cos_to_sin_data(
            d_num=args.sin_data_number,
            data_interval=args.sin_data_interval,
            data_sequence=args.data_sequence_length,
        )

    train_dataset = SinDataset(total_data=total_data, data_type="train")
    valid_dataset = SinDataset(total_data=total_data, data_type="valid")
    test_dataset = SinDataset(total_data=total_data, data_type="test")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    lstm_input_dim = len(train_dataset[0][1][-1])
    models = {
        "SinLSTM": SinLSTM(
            input_dim=lstm_input_dim,
            lstm_hidden_dim=args.hidden_dim1,
            num_layers=args.num_layers,
        ).to(args.device),
        "Cos2SinLSTM": SinLSTM(
            input_dim=lstm_input_dim,
            lstm_hidden_dim=args.hidden_dim1,
            num_layers=args.num_layers,
        ).to(args.device),
    }
    model = models[args.model_name]
    print(model)

    # hyper-params
    loss_funcs = {"L2": nn.MSELoss()}
    criterion = loss_funcs[args.loss]

    lr = args.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )

    save_name = os.path.join(save_dir, f"{save_name}.pth")
    print("*--------Start Training--------*")

    train_valid_evaluate = TrainValidEvaluate(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_epochs=args.n_epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        path=save_name,
        logging=logging,
        # writer=writer
    )
    train_valid_evaluate.sin_train()
    train_valid_evaluate.sin_test()
    # writer.flush()

    logging.info(
        f"TEST SCORE :: \n \ttest MAPE={train_valid_evaluate.avg_test_mape:.4f} \n\ttest loss={train_valid_evaluate.avg_test_loss:.4f}"
    )

    visualize_util = VisualizeUtils(
        predict_value=train_valid_evaluate.test_prediction_result,
        true_value=train_valid_evaluate.test_target_result,
        data_interval=args.sin_data_interval,
    )
    visualize_util.visualize_sin_data("sin prediction result")
    logging.info("========MODEL TRAIN/TEST FINISHED==========\n\n")


if __name__ == "__main__":
    main()
