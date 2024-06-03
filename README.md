![Retip](images/retip_logo.png)

# Retip: Retention Time Prediction for Metabolomics

Retip is a tool for predicting retention times (RTs) of small molecules for high pressure liquid chromatography (HPLC) mass spectrometry.

Retip is a python package for predicting Retention Time (RT) for small molecules in a high pressure liquid chromatography (HPLC) Mass Spectrometry analysis. Retention time calculation can be useful in identifying unknowns and removing false positive annotations. It uses five different machine learning algorithms to built a stable, accurate and fast RT prediction model:

- Random Forest: a decision tree algorithms.
- XGBoost: an extreme Gradient Boosting for tree algorithms.
- AutoGluon: is an automatic machine learning library.
- H2O AutoML: is an automatic machine learning tool.

## Retip installation

Retip requires Python 3.1.0 and it is recommended to use the [Anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html) environments to easily install and run the package.

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
