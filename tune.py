"""
Script for Bayesian hyperparameter tuning.
"""
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


def run_trial(train_data: tt.Dataset, val_data: tt.Dataset,
              device: torch.device, writer, params: dict) -> float:
    global hyperparameter_names
    global trial_ctr
    global all_results

    # Prepare to record results
    trial_results = [params[h] for h in hyperparameter_names]

    # Set up data iterators with trial batch size and bptt length
    batch_size = params["batch_size"]
    bptt = params["bptt"]
    del params["batch_size"]
    del params["bptt"]

    train_iter = tt.BPTTIterator(train_data, batch_size, bptt, device=device,
                                 repeat=False)
    val_iter = tt.BPTTIterator(train_data, 10, bptt, device=device,
                               repeat=False)

    # Run trial
    trial_ctr += 1
    model_name = "trial_{}.pt".format(trial_ctr)
    result = run_experiment(model="LSTM", device=torch.device("cuda"),
                            save=model_name, train_iter=train_iter,
                            test_iter=val_iter, val_iter=val_iter, **params)

    # Record results
    trial_results.append(result)
    all_results.append(trial_results)

    return result


if __name__ == "__main__":
    # Load data
    device = torch.device("cuda")
    train_data, val_data, test_data = load_data("WikiText2", device)

    # Optimize
    params = [dict(name="batch_size", type="range", bounds=[32, 256]),
              dict(name="bptt", type="range", bounds=[16, 128]),
              dict(name="clip", type="range", bounds=[0, 10]),
              dict(name="dropout", type="range", bounds=[0, .75]),
              dict(name="emsize", type="range", bounds=[150, 700]),
              dict(name="lr", type="range", bounds=[10, 50]),
              dict(name="nhid", type="range", bounds=[150, 700])]

    eval_func = lambda p: run_trial(train_data, val_data, device, p)
    optimize(params, eval_func, objective_name="perplexity")

    # Record results
    with open("results.csv", "w") as f:
        results_writer = csv.writer(f)
        results_writer.writerow(hyperparameter_names + ["perplexity"])
        for r in all_results:
            results_writer.writerow(r)
