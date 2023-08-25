![Retip](images/retip_logo.png)

# Retip: Retention Time Prediction for Metabolomics

Retip is a tool for predicting retention times (RTs) of small molecules for high pressure liquid chromatography (HPLC) mass spectrometry.

## Installation and Running Retip

We recommend using the [Anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html) environments to easily install and manage your Python environment.

### Linux/MacOS

Once Anaconda is installed, simply check out the Retip repository and create an environment named `retip`:

```shell
git clone https://github.com/oloBion/pyRetip.git
cd pyRetip
conda env create
```

Now, run `conda activate retip` followed by `jupyter lab`.  You can then open the notebooks folder to start working through code examples to learn how Retip works.

### Windows

Follow the same instructions above and after running `conda activate retip` install the `curses` dependency by running `pip install windows-curses`.

Now run `jupyter lab` and you can then open the notebooks folder to start working through code examples to learn how Retip works.
