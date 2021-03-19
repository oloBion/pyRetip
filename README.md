![Retip](images/retip_logo.png)

# Retip: Retention Time Prediction for Metabolomics

Retip is a tool for predicting retention times (RTs) of small molecules for high pressure liquid chromatography (HPLC) mass spectrometry.

## Installation and Running Retip

We recommend using the [Anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html) environments to easily install and manage your Python environment.  

### Linux/MacOS

Once Anaconda is installed, simply check out the Retip repository and create an environment named `retip`:

```shell
git clone https://github.com/oloBion/pyRetip.git
cd retip2
conda env create
```

Now, run `conda activate retip` followed by `jupyter lab`.  You can then open the notebooks folder to start working through code examples to learn how Retip works.

### Windows

[Download the repository code](https://github.com/oloBion/pyRetip/archive/master.zip) and unextract it or clone the repository using Git for Windows.  You can create the `retip` environment in one of two ways:

1. Open Anaconda Navigator, click on Environment and then Import and select `environment.yml` file in the pyRetip folder.  Click on the arrow next to the new `retip` environment and open a Jupyter Notebook.  Use the navigation in the browser window that opens to find the pyRetip folder, and then open the notebooks folder to start working through code examples to learn how Retip works. 

2. Open the Anaconda Console, navigate to the local Retip directory using `cd`, and run:

```shell
conda env create
activate retip
```

Now run `jupyter lab` and you can then open the notebooks folder to start working through code examples to learn how Retip works.
