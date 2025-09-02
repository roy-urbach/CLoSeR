# Brain-inspired Cooperative Learning of Semantic Representations (CLoSeR)
Code for the paper _Semantic representations emerge in brain-inspired cross-supervising ensembles of neural networks_.

Authors: Roy Urbach & Elad Schneidman

## Links
Soon to be published on arXiv.


## Overview
Cooperative Learning of Semantic Representations (CLoSeR) is a brain-inspired algorithm for 
cross-supervising neural networks for representation learning. 
Similar to cortical columns in the brain, each encoder receives a fixed and random subset of the input. 
Then, they are trained to have similar embeddings of the same input.

![image](images/main_scheme.png)

## Installation
    conda create --name <env> --file requirements.txt
Use conda to create a virtual environment based on [requirements.txt](requirements.txt).


## Repository structure
In this work, the model was evaluated on two modalities - images and neuronal activity.
In the repository, each modality corresponds to a "module", that must have a similar structure:

    CLoSeR/
         └── module
            ├── config/                 # where configurations area saved
            ├── models/                 # where weights and training history are saved
            └── model/         
                ├── losses.py           # optional. If exists, searches for the given loss in this file
                └── model.py            
                    ├── create_model    # a function that inits a model
                    └── compile_model   # a function that compiles a model
            └── evaluate.py
                    └── evaluate        # a function that evaluates a model
            └── utils/
                └── data.py             # with a dataset calss that inherits from ~/utils/utils/data classes
                    └── Labels          # an enum where each object is a utils\data\Label


Therefore, in both modules (vision, neuronal), this structure is kept.
Also, each module has to be inserted to the [utils.modules.Modules](utils/modules.py) enum.

An explanation of the files and folders at the root:

    CLoSeR/
    ├── figures/                        # a directory to save figures in notebooks/figures_notebook.ipynb
    ├── images/                         # a folder with images for the README
    ├── neuronal/                       # a module for the neuronal data and model
    ├── notebooks/                      # notebooks (see section below)
    ├── utils/                          # module-general utils
    ├── vision/                         # a module for the vision model
    ├── .bashrc                         # the bashrc we used, for convenience (especially for IBM LSF clusters)
    ├── cls_likelihood.py               # a script to calculate the mean class-class conditional pseudo-likelihood (figure 2a)
    ├── evaluate.py                     # a script to evaluate a model
    ├── measure.py                      # a script to calculate the cross-path measures of a model
    ├── process_neuronal_dataset.py     # pre-processing needed over the Allen observatory data
    ├── README.md                       # this file
    ├── requirements.txt                # a requirements file, for conda installation 
    ├── run_before_script.py            # a script called by other scripts. Needed for a dynamic adding of methods to the modules
    ├── save_runtime.sh                 # a script that sums the runtime of all jobs that were done to train a model (in IBM LSF clusters)
    ├── train.py                        # a script to train a model
    └── train_cmd_format.py             # a script that prints the command to run to send a job to train a model


## Usage

### Train a model
Use the [train.py](train.py) script to train your model, in the following way:

    python3 train.py -j {model} -m {module} -b {batchsize} -e {epochs}

This assumes _model_ has a corresponding configuration file.
See [train.py](train.py) (and specifically the function _parse_) for details.

If you are using parallelization (and specifically IBM LSF cluster), 
consider using the bash scripts in [.bashrc](./bashrc), for example the function _train_.


You can also use the [training notebook](notebooks/train.ipynb) if it was more convenient for you.


### Evaluate a model
Use the [evaluate.py](evaluate.py) script to evaluate your model, in the following way:

    python3 evaluate.py -j {model} -m {module}

This will save a _classification_eval.json_ in _module_/models/_model_/ with the keys as the names of the metrics, 
and the values as a tuple of (train, validation, test) results.
If you are using parallelization (and specifically IBM LSF cluster), 
consider using the bash scripts in [.bashrc](./bashrc), for example the function _evaluate_.


## Notebooks

We added two notebooks to the repository:

- [figures_notebook](notebooks/figures_notebook.ipynb) - that shows all the calculations done to create the figures in the paper. All the figures will be saved in [figures](figures). 
- [train](notebooks/train.ipynb) - that shows how to train and evaluate a model.
