# DeepPaSTL: Spatio-Temporal Deep Learning Methods for Predicting Long-Term Pasture Terrains using Synthetic Datasets


## Dependencies
* [Install and Setup Anaconda](https://www.anaconda.com/products/individual)

* Clone this repository and `cd` into root folder of the project.

* Create a new conda environment using the dependency file in the project folder.

```
conda env create -f environment.yml
```

* Activate the environment to run the training or testing scripts.
```
conda activate ag-bay-torch
```

## Arguments
To Run the predictions for the first time, or using a new dataset please ensure
that you have the following arguments enabled:

* `--load_data`: Scales and performs necessary pre-processing of the input data for creating final
sliding window inputs for the network.

* `--sequence_data`: Create sliding windows and stores into a **csv** file

* `--sequence_to_np`: Generates numpy arrays and stores in an hdf5 format, for read on access memory, which can be used directly in the Pytorch DataLoader function.

* `--in_seq_len VALUE`: Input Sequence Length. Expects an **int** value.
* `--out_seq_len VALUE`: Output Sequence Length. Expects an **int** value.

* `--device cpu`: To run inference or training on CPU, else default is at `cuda`.

* `--exp_name MODEL_NAME`: Experiment name to load the model during inference, or save the mode as during training.

## Prediction/Testing
```
python main.py --in_seq_len IN_VALUE --out_seq_len OUT_VALUE --load_data --sequence_data --sequence_to_np
```

## Training
```
python main.py --train_network --in_seq_len IN_VALUE --out_seq_len OUT_VALUE --load_data --sequence_data --sequence_to_np
```

## Output
Output is stored as a MATLAB file in `.mat` file. Data is stored in the following dictionaries.

* `date`: Start date of the first day of the sequence. That is, the first date of the start of the input sequence.
* `y_pred_mean`: Mean prediction/inference output from different prediction samples: `--n_samples VALUE`. Dimension is `(batch, out_seq_len, x_dim, y_dim)`
* `y_std_mean`: Standard Deviation of the prediction/inference output from different prediction samples: `--n_samples VALUE`. Dimension is `(batch, out_seq_len, x_dim, y_dim)`
* `y_target`: Ground truth/Target values for the predicted output. Dimension is `(batch, out_seq_len, x_dim, y_dim)`
