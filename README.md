# Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivity

This repository provides a python implementation of machine learning approach described in our TMI paper [Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivity.](https://ieeexplore.ieee.org/document/9874890)

## Download required packages

```bash
pip install -r requirements.txt
```

## Download and preprocess ABIDE data

In the files ``imports/train.py``, ``imports/preprocess_data.py`` and ``fetch_data.py``, change 'path/to/data/' to an appropriate file path.
To download the ABIDE data, run:

```bash
python fetch_data.py --cfg configs/download_abide.yaml
```

Options are available for the preprocessing pipeline, brain atlas and functional connectivity. Please see the `config.py` for information about the available options. The .yaml files under the `./configs` folder are examples to specify the options.

## Classification

The default model provided is the MIDA model with tangent Pearson functional connectivity + phenotypes trained with a ridge classifier and evaluated with 10 fold cross validation (CV). Can be run by:

```bash
python run_model.py --cfg/run_default.yaml
```

The default functional connectivity here is tangent Pearson embedding and is computed at run time separately for train and test folds.

### Citation

```lang-latex
    @article{kunda2022improving,
      title={Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivity},
      author={Kunda, Mwiza and Zhou, Shuo and Gong, Gaolang and Lu, Haiping},
      journal={IEEE Transactions on Medical Imaging},
      year={2022},
      publisher={IEEE},
      doi={10.1109/TMI.2022.3203899}
    }
```
