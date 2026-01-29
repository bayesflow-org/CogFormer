import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def prior_predictive(
    design_config: dict,
    intrinsic_params: list[str],
    variable_names: list[str],
    figsize: tuple[int, int],
):
    num_params = len(intrinsic_params)
    if figsize is None:
        figsize = (3, 3 * num_params)

    fig, ax = plt.subplots(1, num_params, figsize=figsize)
    return fig


if __name__ == "__main__":

    intrinsic_params = ["v", "a", "z", "tau"]
    regressed_params = ["v", "a"]

    design_config = {
        "1": intrinsic_params,
        "u_1": regressed_params,
        "u_2": regressed_params,
        "u_1:u_2": regressed_params,
    }

    fig = prior_predictive(
        design_config = design_config,
    )
    fig.savefig("test_predictive.pdf")
    print("groovy")