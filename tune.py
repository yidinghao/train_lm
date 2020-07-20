"""
Script for Bayesian hyperparameter tuning.
"""
import argparse
import csv

import torch
from ax.service.managed_loop import optimize
from torchtext import data as tt

from data import load_data
from train import run_experiment

hyperparameter_names = ["batch_size", "bptt", "clip", "dropout",
                        "emsize", "lr", "nhid"]
trial_ctr = 0
all_results = []


def run_trial(model_type: str, train_data: tt.Dataset, val_data: tt.Dataset,
              device: torch.device, prefix: str, params: dict) -> float:
    """
    Runs one trial of hyperparameter optimization

    :param model_type: Model architecture
    :param train_data: Training dataset
    :param val_data: Validation dataset
    :param device: CUDA or CPU
    :param prefix: Unique prefix added to output file names
    :param params: Trial hyperparameters
    :return: The perplexity
    """
    global hyperparameter_names
    global trial_ctr
    global all_results

    # Prepare to record results
    trial_ctr += 1
    trial_results = [params[h] for h in hyperparameter_names]
    print("_" * 69)
    print("Trial", trial_ctr)
    for h in hyperparameter_names:
        print(h, ": ", params[h], sep="")
    print("‾" * 69)

    # Set up data iterators with trial batch size and bptt length
    batch_size = params["batch_size"]
    bptt = params["bptt"]
    del params["batch_size"]
    del params["bptt"]

    try:
        train_iter = tt.BPTTIterator(train_data, batch_size, bptt,
                                     device=device, repeat=False)
        val_iter = tt.BPTTIterator(val_data, 10, bptt, device=device,
                                   repeat=False)

        # Run trial
        model_name = "{}trial_{}.pt".format(prefix, trial_ctr)
        result = run_experiment(model=model_type, device=torch.device("cuda"),
                                save=model_name, train_iter=train_iter,
                                test_iter=val_iter, val_iter=val_iter,
                                epochs=50, **params)
    except RuntimeError:  # Keep going if CUDA runs out of memory
        print("ERROR")
        result = 1e10

    # Record results
    trial_results.append(result)
    all_results.append(trial_results)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--data", type=str, default="WikiText2",
                        help="Dataset: WikiText2 or WikiText103")
    parser.add_argument("--model", type=str, default="LSTM",
                        help="Network architecture: RNN_TANH, RNN_RELU, LSTM, "
                             "GRU, or Transformer")
    parser.add_argument("--jobname", type=str, default="",
                        help="Unique prefix added to output file names")
    args = parser.parse_args()

    # Load data
    device = torch.device("cuda")
    train_data, val_data, test_data = load_data(args.data, device)

    # Optimize
    prefix = "" if args.jobname == "" else args.jobname + "_"
    params = [dict(name="batch_size", type="range", bounds=[32, 256]),
              dict(name="bptt", type="range", bounds=[16, 64]),
              dict(name="clip", type="range", bounds=[0, 10]),
              dict(name="dropout", type="range", bounds=[0., .75]),
              dict(name="emsize", type="range", bounds=[150, 700]),
              dict(name="lr", type="range", bounds=[10, 50]),
              dict(name="nhid", type="range", bounds=[150, 700])]

    eval_func = lambda p: run_trial(args.model, train_data, val_data,
                                    device, prefix, p)
    optimize(params, eval_func, objective_name="perplexity", minimize=True)

    # Record results
    with open(prefix + "results.csv", "w") as f:
        results_writer = csv.writer(f)
        results_writer.writerow(hyperparameter_names + ["perplexity"])
        for r in all_results:
            results_writer.writerow(r)
