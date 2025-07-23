# Semantic representations emerge in brain-inspired cross-supervising ensembles of neural networks
Roy Urbach & Elad Schneidman

Code used for the above paper.


## Links
Awaiting NeurIPS decision, then it will be published on arXiv.


## Overview
Cooperative Learning of Semantic Representations (CLoSeR) is a brain-inspired algorithm for 
cross-supervising neural networks for representation learning. 
Similar to cortical columns in the brain, each encoder receives a fixed and random subset of the input. 
Then, they are trained to have similar embeddings of the same input.


<img src="images/main_scheme.png">


## Installation
Use pip or conda to create a virtual environment based on [requirements.txt](requirements.txt).


## Repository structure
In this work, the model was evaluated on two modalities - images and neuronal activity.
In the repository, each modality corresponds to a "module", that must have a similar structure:
>- **config**           (package; where the models configurations files are saved)
>- **models**           (package; where the weights and training history are saved)
>- **model\model.py**   (file)
>  - **create_model**   (function; inits a model)
>  - **compile_model**  (function; compiles a model)
>- **evaluate.py**
>  - **evaluate**        (a function that evaluates a model)
>- **utils\data.py**  (file; with a class that inherits from _~\utils\utils\data_ classes)

Therefore, in both modules (**vision**, **neuronal**), this structure is kept.
Also, each module has to be inserted to the [utils.modules.Modules](utils/modules.py) enum.

## Usage

### Train a model
Use the [train.py](train.py) script to train your model, in the following way:

    python3 train.py -j {model} -m {module} -b {batchsize} -e {epochs}

This assumes _model_ has a corresponding configuration file.
See [train.py](train.py) (and specifically the function _parse_) for details.

If you are using parallelization (and specifically IBM LSF cluster), 
consider using the bash scripts in [.bashrc](./bashrc), for example the function _train_.


You can also use the notebook in _notebooks.train.ipynb_ if it was more convenient for you.


### Evaluate a model
Use the [evaluate.py](evaluate.py) script to evaluate your model, in the following way:

    python3 evaluate.py -j {model} -m {module}

This will save a _classification_eval.json_ in *module*/models/_model_/ with the keys as the names of the metrics, 
and the values as a tuple of (train, validation, test) results.
If you are using parallelization (and specifically IBM LSF cluster), 
consider using the bash scripts in [.bashrc](./bashrc), for example the function _evaluate_.


## Notebooks

We added two notebooks to the repository:

- [figures_notebook](figures_notebook.ipynb) - that shows all the calculations done to create the figures in the paper. All the figures will be saved in [figures](figures). 
- [train](train.ipynb) - that shows how to train and evaluate a model.
