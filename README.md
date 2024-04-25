## Neural Decoder Experiments

Fork of [Neural Sequence Decoder](https://github.com/cffan/neural_seq_decoder), adding an implementation of the [Mamba model](https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file) and [MVTS Transformer model](https://github.com/gzerveas/mvts_transformer) as alternative Neural Decoders.
Also adds [3-gram-only decoding](./scripts/eval_competition_3gram.py), which decodes a given phoneme probability sequence provided by a Neural Sequence Decoder output using only Viterbi decoding. Here the 3 gram model provides the character emission scores whereas the phoneme probabilities serve as the transition scores.
This is the same of the other decoding methods, but they add an LLM for rescoring candidate transcriptions and can use a 5 gram lm instead.

## Prerequisites
- Linux
- Anaconda/Miniconda ([installation](https://docs.anaconda.com/free/anaconda/install/linux/))

## Installation
As we could not train the Mamba model in the environment specified by the original repo, nor install the packages for lm decoding into a functioning train environment
, we introduce a new environment, meaning 2 environments need to be installed.
1. Install `train_b2t` environment: `conda env create --file=train_environment.yml`
2. Install `lm_decoder` environment (environment of the [original repo](https://github.com/cffan/neural_seq_decoder)):
    - Create conda env: `conda create --name lm_decoder python=3.9 `
    - Activate env: `conda activate lm_decoder`
    - Install packages: `pip install -e .`
## How to run

1. With either environment activated, ccnvert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. With `train_b2t` env active, train model: `python ./scripts/train_model.py`

