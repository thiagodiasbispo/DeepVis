from pathlib import Path
import numpy as np

# base_path = Path("../Crowded-Valley---Results/results/")
base_path = Path("../Crowded-Valley---Results/results_main/")


# All possible metrics
metrics_cols = [
    "train_losses",
    "valid_losses",
    "test_losses",
    "train_accuracies",
    "valid_accuracies",
    "test_accuracies",
    "minibatch_train_losses",
]

params_cols = ["budget", "schedule", "testproblem", "optimizer", "best_params", "seed"]


budgets = ["medium_budget", "oneshot", "large_budget", "small_budget"]

schedules = ["none", "cosine", "cosine_wr", "ltr"]
testproblems = [
    "cifar10_3c3d",
    "cifar100_allcnnc",
    "fmnist_2c2d",
    "mnist_vae",
    "svhn_wrn164",
    "fmnist_vae",
    "quadratic_deep",
    "tolstoi_char_rnn",
]

optimizers = [
    "AMSBoundOptimizer",
    "AMSGrad",
    "AdaBeliefOptimizer",
    "AdaBoundOptimizer",
    "AdadeltaOptimizer",
    "AdagradOptimizer",
    "AdamOptimizer",
    "LookaheadOptimizerMBGDMomentum",
    "LookaheadOptimizerRAdam",
    "MomentumOptimizer",
    "NAGOptimizer",
    "NadamOptimizer",
    "RAdamOptimizer",
    "RMSPropOptimizer",
    "GradientDescentOptimizer",
]

optimizers_to_small_name_dict = {
    "AMSBoundOptimizer": "AMSBound",
    "AMSGrad": "AMSGrad",
    "AdaBeliefOptimizer": "AdaBelief",
    "AdaBoundOptimizer": "AdaBound",
    "AdadeltaOptimizer": "Adadelta",
    "AdagradOptimizer": "Adagrad",
    "AdamOptimizer": "Adam",
    "LookaheadOptimizerMBGDMomentum": "LA(Mom.)",
    "LookaheadOptimizerRAdam": "LA(RAdam)",
    "MomentumOptimizer": "Mom.",
    "NAGOptimizer": "NAG",
    "NadamOptimizer": "Nadam",
    "RAdamOptimizer": "RAdam",
    "RMSPropOptimizer": "RMSProp",
    "GradientDescentOptimizer": "SGD",
}

testproblems_loss_type = {t: "test_accuracies" for t in testproblems}

testproblems_loss_type.update(
    {"fmnist_2c2d": "test_losses", "mnist_vae": "test_losses"}
)


def to_small_name(names):
    return [optimizers_to_small_name_dict[n] for n in names]


optimizers_small_name = to_small_name(optimizers)


def coalesce(acc, loss):
    return round(acc * 100, 2) if not np.isnan(acc) else round(loss, 2)
