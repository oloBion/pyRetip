![Retip](images/retip_logo.png)

# Retip: Retention Time Prediction for Metabolomics

Retip is a tool for predicting retention times (RTs) of small molecules for high pressure liquid chromatography (HPLC) mass spectrometry.

## Installation

We recommend using the [Anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html) environments to easily install and manage your Python environment.  

### Linux/MacOS

Once Anaconda is installed, simply check out the Retip repository and create the environment:

```shell
git clone https://github.com/oloBion/retip2.git
cd retip2
conda env create -f micasense_conda_env.yml
```

### Windows

[Download the repository code](https://github.com/oloBion/retip2/archive/master.zip) and unextract it or clone the repository using Git for Windows.  Then open the Anaconda Console, navigate to the local Retip directory using `cd`, and run:

```shell
conda env create -f micasense_conda_env.yml
```

## Running Retip

After setting up your environment, run `conda activate retip` in Linux/MacOS or `activate retip` in Windows.  Then run `jupyter lab` and open the notebooks folder to start working through code examples to learn how Retip works
