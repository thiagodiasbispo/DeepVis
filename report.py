import pandas as pd
import numpy as np
from common import optimizers_small_name


def get_optimized_vs_untuned_results(
    df: pd.DataFrame,
    testproblem: str,
    other_budget: str,
    schedule: str = "none",
    metric_col: str = "test_accuracies",
    return_adj_matrix: bool = True,
):

    df2 = df[
        (df.testproblem == testproblem)
        & (df.budget.isin([other_budget, "oneshot"]))
        & (df.schedule == schedule)
        & (df.best_params == True)
    ]
    if return_adj_matrix:
        improvement = np.zeros((len(optimizers_small_name), len(optimizers_small_name)))
    else:
        improvement = []

    for i, opt1 in enumerate(optimizers_small_name):
        oneshot = df2.loc[(df2.optimizer == opt1) & (df2.budget == "oneshot")][
            metric_col
        ].mean()

        for j, opt2 in enumerate(optimizers_small_name):

            other = df2.loc[(df2.optimizer == opt2) & (df2.budget == other_budget)][
                metric_col
            ].mean()
            val = (other - oneshot) * 100
            if return_adj_matrix:
                improvement[i, j] = val
            else:
                improvement.append((opt2, opt2, round(val, 2)))

    improvement = improvement.round(2)
    if return_adj_matrix:
        return pd.DataFrame(
            improvement, columns=optimizers_small_name, index=optimizers_small_name
        )
    else:
        return pd.DataFrame(
            improvement,
            columns=["optimizer1", "optimizer2", "improvement"],
            index=optimizers_small_name,
        )
