import argparse

import torch
import torch.nn.parallel
from torch.utils.data import DataLoader

import utils
from model import GMADE

if __name__ == "__main__":
    # Set seeds for reproducibility
    utils.set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/Dec08_10-19-53_code-server/model.pt", help="Path to saved model from training")
    model_dir = parser.parse_args().model

    # Load model state dict and args
    saved_model = torch.load(model_dir)
    state_dict, args = saved_model['state_dict'], saved_model['args']

    # Load data
    _, val_loader, test_loader = utils.load_data(args.dataset, args.bs)

    # Initialize the model and optimizer
    model = GMADE(13 * 5, args.hs, 13 * 5 * 2).to(args.device)

    # Evaluation model generalization using test set

    roc_auc, balacc, tpr, tnr = utils.test_model(model, val_loader, test_loader, state_dict, args)

    print('ROC-AUC:', roc_auc)
    print('BALACC:', balacc)
    print('TPR:', tpr)
    print('TNR:', tnr)