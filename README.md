# Solution for MIDST

## Introduction

This repository is mainly about a solution for the MIDST white-box single-table task. My method is based on a white-box membership inference attack approach called **GSA**, introduced in the paper [*White-box membership inference attacks against diffusion models*](https://petsymposium.org/popets/2025/popets-2025-0068.pdf), accepted by PoPets 2025. The core idea is to leverage the differences in gradients between member and non-member samples for the attack.

The solution contains the following steps:

1. Using the data under the `train` path to extract gradients of member and non-member samples at various timesteps. In this scenario, the models under the `train` path are treated as **shadow models**.
2. For each sample’s extracted data, we perform processing, such as selecting the most significant gradient dimensions (since gradient data is high-dimensional, a simple approach is to compute the norm of the gradient for each layer).
3. Using the extracted gradient data to train an **attack model**.
4. Testing the attack model with the models under the `dev` and `final` paths, and then generating `prediction.csv`.

> Due to task requirements, I have modified the code in `midst_models/single_table_TabDDPM` related to model training and data processing.


## Environment Setup

My solution does not require additional packages. You can fully follow the environment setup process from the original repository:

1. Clone the original [GitHub repository](https://github.com/VectorInstitute/MIDSTModels) to your local environment.
2. Run the following commands:

    ```bash
    pip install --upgrade pip poetry
    poetry env use [name of your python] # e.g., python3.9
    source $(poetry env info --path)/bin/activate
    poetry install --with "tabsyn, clavaddpm"
    # If your system is not compatible with pykeops, you can uninstall it using the following command:
    pip uninstall pykeops
    ```

## Attack Process

1. **Extract gradients**: Run `get_gradients.py` to extract gradient data of the `train` path models for both member and non-member samples. This script will save the gradient data in `train_data.pt` and the corresponding labels in `train_label.pt`.

    ```bash
    python get_gradients.py
    ```

2. **Train the attack model**: Run `train_model.py` to train a classification model (the attack model) on the extracted gradient data. This script will log scalars and save checkpoints that achieve both train accuracy and test accuracy above 0.70.

    ```bash
    python train_model.py
    ```

3. **Evaluate on dev and final**: Run `evaluate_with_dev.py` to load the best-performing attack model and the corresponding scalar, then apply it to the models in the `dev` and `final` paths, generating `prediction.csv`.

    ```bash
    python evaluate_with_dev.py
    ```

## Acknowledgements

I would like to express my gratitude to the Vector Institute for organizing this competition. It has been a valuable opportunity to further explore membership inference attacks against diffusion models. I also apologize for the lack of comments in the code. If you are interested in my solution, feel free to email me at `trv3px@virginia.edu` for more information.
