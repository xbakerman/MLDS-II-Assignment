# MLDSII - Assignment 
## Spamfilter

This is the repository for the Machine Learning & Data Science 2 final Assignment. 
The goal is to create a spam filter for emails.
The datasets are from the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/) and are in the 'Datasets' folder.
The source code is in the 'main.py' file.
The report with explanation of the code and decisions is in the 'SpamFilter_Report_Nachbaur_Maximilian.pdf' file.

## Setup

For this project I used:
- [python](https://www.python.org/) as programming language
- [PDM](https://pdm-project.org/latest/) for the `python` package management

### Python

The Version is `python` 3.12.* for this project.
For the complete dependencies see [pyproject.toml](./pyproject.toml).

### PDM
To install `pdm` run
```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

#### Initialize
The `pdm` project already exists so all you need to do is to run 
```bash
pdm sync
```
or 
```bash
pdm install
```
to initialize the project.

#### Start a script
To start a script use
```bash
pdm run python <SCRIPT.py> <arguments>
```

You can also start a `python` console via
```bash
pdm run python
```
Note, that both will respect the `.env` file and load the variables accordingly.
