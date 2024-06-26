![Retip](images/retip_logo.png)

# Retip - Retention Time Prediction for Metabolomics

Retip 2.0 was updated and released in June 2024 by [oloBion](https://www.olobion.ai/).

## Introduction

**Retip** is a tool for predicting Retention Time (RT) for small molecules in a high pressure liquid chromatography (HPLC) Mass Spectrometry analysis, available as both a [**Python package**](https://github.com/oloBion/pyRetip/tree/master) and an [**R package**](https://github.com/olobion/Retip/tree/master). Retention time calculation can be useful in identifying unknowns and removing false positive annotations. The [**Python package**](https://github.com/oloBion/pyRetip/tree/master) uses four different machine learning algorithms to built a stable, accurate and fast RT prediction model:

- **Random Forest:** a decision tree algorithms.
- **XGBoost:** an extreme Gradient Boosting for tree algorithms.
- **AutoGluon:** is an automatic machine learning library.
- **H2O AutoML:** is an automatic machine learning tool.

## Retip installation

Retip 2.0 requires Python 3.10 and it is recommended to use the [Anaconda](https://www.anaconda.com/download/), [miniconda](https://conda.io/miniconda.html) or [Mamba](https://mamba.readthedocs.io) package managers to easily install and run the package.

Once Anaconda is installed, simply check out the Retip repository and create an environment named `pyretip`:

```shell
git clone https://github.com/oloBion/pyRetip.git
cd pyRetip
conda env create
```

### Linux/MacOS

Run `conda activate pyretip` followed by `jupyter lab`. Then, open the [notebooks folder](https://github.com/oloBion/pyRetip/tree/master/notebooks) to start working through code examples to learn how Retip works.

### Windows

Run `conda activate pyretip` and install the `curses` dependency by running `pip install windows-curses`. Then, run `jupyter lab` and open the [notebooks folder](https://github.com/oloBion/pyRetip/tree/master/notebooks) to start working through code examples to learn how Retip works.
