# Adjusting for the progressive digitization of health records: working examples on a multi-hospital clinical data warehouse


<div align="center">
    <img src="logo.svg" alt="EDS-TeVa">

<p align="center">
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org/" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-poetry-blue" alt="Poetry">
</a>
<a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-%3E%3D%203.7.1%20%7C%20%3C%203.8-brightgreen" alt="Supported Python versions">
</a>
<a href="https://spark.apache.org/docs/2.4.8/" target="_blank">
    <img src="https://img.shields.io/badge/spark-2.4-brightgreen" alt="Supported Java version">
</a>
</p>
</div>

## Study

This repositoy contains the computer code that has been executed to generate the results of the article:
```
@unpublished{edsteva,
author = {Adam Remaki and Benoît Playe and Paul Bernard and Simon Vittoz and Matthieu Doutreligne and Gilles Chatellier and Etienne Audureau and Emmanuelle Kempf and Raphaël Porcher and Romain Bey},
title = {Adjusting for the progressive digitization of health records: working examples on a multi-hospital clinical data warehouse},
note = {Manuscript submitted for publication},
year = {2023}
}
```
The code has been executed on the OMOP database of the clinical data warehouse of the  <a href="https://eds.aphp.fr/" target="_blank">Greater Paris University Hospitals</a>

- IRB number: CSE210033
- This study stands on the shoulders of the library [EDS-TeVa](https://github.com/aphp/edsteva) (an open-source library providing a set of tools that aims at modeling the adoption over time and across space of the Electronic Health Records).
## Version 1.0.0

- Submission of the article for review.
## Setup

- In order to process large-scale data, the study uses [Spark 2.4](https://spark.apache.org/docs/2.4.8/index.html) (an open-source engine for large-scale data processing) which requires to:

  - Install a version of Python $\geq 3.7.1$ and $< 3.8$.
  - Install Java 8 (you can install [OpenJDK 8](https://openjdk.org/projects/jdk8/), an open-source reference implementation of Java 8)

- Clone the repository:

```shell
git clone https://gitlab.eds.aphp.fr/equipedatascience/cse_210033.git
```

- Create a virtual environment with the suitable Python version (**>= 3.7.1 and < 3.8**):

```shell
cd cse_210033
python -m venv .venv
source .venv/bin/activate
```

- Install [Poetry](https://python-poetry.org/) (a tool for dependency management and packaging in Python) with the following command line:
    - Linux, macOS, Windows (WSL):

    ```shell
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    - Windows (Powershell):

    ```shell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

    For more details, check the [installation guide](https://python-poetry.org/docs/#installation)

- Install dependencies:

```shell
pip install pypandoc==1.7.5
pip install pyspark==2.4.8
poetry install
pip uninstall pypandoc
```
## How to run the code on AP-HP's data platform
### 1. Install EDS-Toolbox:

EDS-Toolbox is python library that provides an efficient way of submitting PySpark scripts on AP-HP's data platform. As it is AP-HP specific, it is not available on PyPI:

```shell
pip install git+ssh://git@gitlab.eds.aphp.fr:2224/datasciencetools/edstoolbox.git@v0.6.3
```
### 2. Pre-processing: Compute, save models and data:

:warning: Depending on your resources, this step can take some times.

```shell
cd scripts
eds-toolbox spark submit --config ../conf/config.cfg --log-path ../logs/ehr_modeling ehr_modeling.py
eds-toolbox spark submit --config ../conf/config.cfg --log-path ../logs/cohort_selection cohort_selection.py
```

### 3. Post-processing: Main statistical analysis

```shell
pip install pyarrow==12.0.1
python statistical_analysis.py --config-name config.cfg
```

### 4. Generate figures

- **Option 1**: Generate all figures in a raw from the terminal:

  ```shell
  python generate_figures.py --config-name config.cfg
  ```

- **Option 2**: Generate figure one at a time from a notebook:

  - Create a Spark-enabled kernel with your environnement:

    ```shell
    eds-toolbox kernel --spark --hdfs
    ```

   - Convert markdown into jupyter notebook:

      ```shell
      cd notebooks
      jupytext --set-formats md,ipynb 'generate_figures.md'
      ```

   - Open *generate_figures.ipynb* and start the kernel you've just created.
     - Run the cells to obtain every figure.

### 5. Generate HTML report

- Create a Spark-enabled kernel with your environnement (if you have not previously):

```shell
eds-toolbox kernel --spark --hdfs
```

- Convert markdown into jupyter notebook:

```shell
cd notebooks
jupytext --set-formats md,ipynb 'report.md'
```

- Open *report.ipynb*, start the kernel you've created and run the cells.

- Convert notebook to HTML:
```shell
eds-toolbox report report.ipynb --output report.html
```

#### Note
If you would like to run the scripts on a different database from the AP-HP database, you will have to adapt the python scripts with the configuration of the desired database.
## Project structure

- `conf`: Configuration files.
- `data`: Saved processed data and models.
- `figures`: Saved results.
- `notebooks`: Notebooks that generate figures.
- `cse_210033`: Source code.
- `scripts`: Typer applications to process data and generate figures.

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
