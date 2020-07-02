# u-MPS
u-MPS implementation and experimentation code used in the paper [Tensor Networks for Probabilistic Sequence Modeling](https://arxiv.org/abs/2003.01039).

This does not include the full regex sampler described in the paper.

## Dependencies

Jax, Pytorch, and Numpy should be installed and accessible for import within Python.

## How to Run the Experiments

All experiment scripts are included in the experiments folder, and running `tomita_exp.py` or `motzkin_exp.py` will train u-MPS and LSTM models in the manner described in our paper. These scripts will save the trained models and experimental data from each experiment, and the scripts `tomita_resample.py` and `motzkin_resample.py` can then be used to obtain sampling statistics for strings of different lengths using the trained models.

Although training the models prints a lot of information to stdout, the scripts `tomita_info.py` and `motzkin_info.py` can be used after training to output only the relevant high-level statistics for the experiment.

To ensure the same trained u-MPS model is used for both the Motzkin completion and sampling tasks, `motzkin_exp.py` can be run with `comp_exp = False`, and sampling statistics for the completion task then obtained by running `motzkin_resample.py` with `comp_exp = True`.

The trained models and data used for our experiments are contained in the save files `.tomita_exp_paper.record`, `.motzkin_exp_paper.record`, and `.motzkin_exp_comp_paper.record`. The numbers reported in our tables can be obtained via `tomita_info.py` and `motzkin_info.py`, with `save_name` set to the corresponding save file.