This code provides a python implementation of the second-order functional connectivity feature and domain adaptation approach described in our paper [Improving multi-site autism classification based on site-dependence minimisation and second-order functional connectivity for Autism classification on the ABIDE dataset.](https://www.biorxiv.org/content/10.1101/2020.02.01.930073v1)

## Download required packages
```
pip install -r requirements.txt
```
## Download and preprocess ABIDE data
In the files ``imports/train.py``, ``imports/preprocess_data.py`` and ``fetch_data.py``, change 'path/to/data/' to an appropriate file path.
To download the ABIDE data, run:
```
python fetch_data.py
```
Options are available for the preprocessing pipeline, brain atlas and functional connectivity. Run `python fetch_data.py --help` for information about the available options. For tangent Pearson connectivity, see below.

## Classification
The default model provided is the MIDA model with tangent Pearson functional connectivity + phenotypes trained with a ridge classifier and evaluated with 10 fold cross validation (CV). Can be run by:
```
python run_model.py
```
The default functional connectivity here is tangent Pearson embedding and is computed at run time separately for train and test folds. As above, run `python run_model.py --help` to see the available options for the choice of brain atlas, functional connectivity, evaluation method (e.g. 10CV, leave one site out) e.t.c.
