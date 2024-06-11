<h1 align="center">KOREGRAPH</h1>

_<h4 align="center">An IA for music driven choregraphy generation.</h4>_

<div align="center">

[![Pipeline status](https://github.com/Anatole-DC/koregraph/actions/workflows/base.yml/badge.svg)](https://github.com/Anatole-DC/koregraph/actions)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Gitmoji](https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg)](https://gitmoji.carloscuesta.me/)
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

</div>

___

**Technos :** Python, Poetry

**Author :** [Anatole-DC](https://github.com/Anatole-DC)

___

## Summary

- [Summary](#summary)
- [Requirements](#requirements)
  - [Git](#git)
  - [Python 3.8.10](#python-3810)
  - [Poetry](#poetry)
  - [VSCode (Optional)](#vscode-optional)
- [Install the project](#install-the-project)
  - [Clone the repository](#clone-the-repository)
  - [Install the dependencies](#install-the-dependencies)
- [Get a prediction](#get-a-prediction)
- [Scripts](#scripts)
- [Ideas](#ideas)
- [License](#license)

## Requirements

### [Git](https://git-scm.com/)

```bash
git --version
# git version 2.34.1
```

### [Python 3.8.10](https://www.python.org/)

```bash
python --version
# Python 3.10.12
```

### [Poetry](https://python-poetry.org/)

```bash
poetry run --version
# Poetry (version 1.6.1)
```

<details>
  <summary>If you are using pyenv</summary>

**Activate koregraph**

```bash
pyenv local koregraph
```

**Install poetry**

```bash
pip intall poetry
```

</details>

### [VSCode (Optional)](https://code.visualstudio.com/)

This template is configured to work with the VSCode editor, but it does not required it to be used.

## Install the project

### Clone the repository

```bash
git clone https://github.com/Anatole-DC/koregraph
cd koregraph
```

### Install the dependencies

```bash
poetry shell
poetry install
```

**In dev mode :**

```bash
poetry install --with dev,viewer
```

## Get a prediction
**Generate features and output pkl files**
```bash
poetry run generate
```
**Train a model**
```bash
poetry run train -m model_kelly
```
**Get a video**
```bash
poetry run predict -m model_kelly -a mBR0
```

## Scripts

**Export presentation (github pages)**

```bash
poetry run jupyter nbconvert --no-input frontend/views/presentation.ipynb  --to slides --stdout > documentation/pages/index.html
```

**Train a model**

```bash
poetry run train --help

# Pass the -d option to only take a random sample of the dataset
poetry run train -m modelkelly -d 0.5

# Train on the cloud (ensure the environment variable are set)
poetry run train -m modelkelly --with-cloud -d
```

**Predict a choregraphy from an audio**

```bash
poetry run predict -a mBR1 -m model -i 01
```

**Build viewer video from choregraphy**

```bash
poetry run viewer -c data/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl  # Path to your video
```

**Create 5 sec chunks for a choreography**

```bash
poetry run chunk -c data/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch02.pkl -s 5
```
If the music already exists, but you want to split it again for some reason, add the following parameter `--reload-music`

## Ideas

Pitch en ligne

Transformer mercredi deuxième semaine

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
