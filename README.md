# Reimplementation of Deep One-Class Classification

## Disclosure
This implementation is based on the repository 
[https://github.com/lukasruff/Deep-SVDD](https://github.com/lukasruff/Deep-SVDD), which is in turn based on [https://github.com/oval-group/pl-cnn](https://github.com/oval-group/pl-cnn), which is licensed under the MIT license. 

The *pl-cnn* repository is an implementation of the paper 
[Trusting SVM for Piecewise Linear CNNs](https://arxiv.org/abs/1611.02185) by Leonard Berrada, Andrew Zisserman and 
M. Pawan Kumar, which was an initial inspiration for this research project.

You find the PDF of the Deep One-Class Classification ICML 2018 paper at 
[http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html).

If you intend to use this work, please look at also look at the original implementation [https://github.com/lukasruff/Deep-SVDD](https://github.com/lukasruff/Deep-SVDD) first, since this reimplentation is not as well structured as the original repository.

Also cite the original Deep SVDD ICML 2018 paper:
```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4390--4399},
  year      = {2018},
  volume    = {80},
}
```

## Repository structure

### `/data`

Contains the data for the original DSVDD experiments. To add new experiments, you will have to explicitly specify paths to your data in `src/config.py` so you can store it anywhere on your system.

### `/src`
Source directory that contains all Python code and shell scripts to run experiments.

### `/log`
Directory where the results from the experiments are saved.


## To reproduce results
Change your working directory to `src` and make sure the settings you want are specified in config.py. The Deep SVDD implementation uses shell scripts with command line arguments to run experiments. 

For the .sh-files in `src/scripts`, these arguments need to be specified. It will call the main file, `baseline.py` with the specified arguments.

In the .sh files in `src/experiments`, there are scripts with predefined arguments for each of the two implemented SMILE II datasets (Dreyeve and ProSiVIC) and each of the two Deep SVDD methods, soft-boundary and one-class.


To run an experiment with specified arguments, do
```
sh scripts/any_smile_dataset.sh ${dataset} ${device} ${seed} ${solver} ${lr} ${n_epochs} ${nu} ${hard_margin} ${center_fixed} ${block_coordinate} ${in_name} ${batch_size} ${weight_dict_init}
```
For example to run a Prosivic experiment, do

```
sh scripts/any_smile_dataset_svdd.sh prosivic gpu exp_name 0 adam 0.0001 150 1 1 1 0 sunny_highway 64 0;

```
This is also the contents of the script `experiments/prosivic_oneclass_dsvdd.sh`. It runs a *one-class Deep SVDD* (`hard_margin = 1` and `block_coordinate = 0`) experiment where one-class Deep SVDD is trained for `n_epochs = 150` with the Adam optimizer 
(`solver = adam`) and a learning rate of `lr = 0.0001`. The experiment is executed on `device = gpu`. The results are stored in `log/prosivic/exp_name/`.

You find descriptions of the various script options within the respective Python files that are called by the shell
scripts (e.g. `baseline.py`).

An addition in this reimplementation is a checkpointing system, where pretrained autoencoders and DSVDD networks are checkpointed regularly, and if training is interrupted at any point, the `baseline.py` script will, when called again with the same arguments, look for the latest appropriate checkpoint and resume training.
