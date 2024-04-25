# Neural Decoder Experiments

Fork of [Neural Sequence Decoder](https://github.com/cffan/neural_seq_decoder), adding an implementation of the [Mamba model](https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file) and [MVTS Transformer model](https://github.com/gzerveas/mvts_transformer) as alternative Neural Decoders.
Also adds [3-gram-only decoding](./scripts/eval_competition_3gram.py), which decodes a given phoneme probability sequence provided by a Neural Sequence Decoder output using only Viterbi decoding (based on code from [Neural Sequence Decoder](https://github.com/cffan/neural_seq_decoder)). Here the 3 gram model provides the character emission scores whereas the phoneme probabilities serve as the transition scores.
This is the same of the other decoding methods, but they add an LLM for rescoring candidate transcriptions and can use a 5 gram lm instead.

## Prerequisites
- Linux
- Anaconda/Miniconda ([installation](https://docs.anaconda.com/free/anaconda/install/linux/))

## Installation
As we could not train the Mamba model in the environment specified by the original repo, nor install the packages for lm decoding into a functioning train environment, we introduce a new environment, meaning 2 environments need to be installed.
1. Install `train_b2t` environment: `conda env create --file=train_environment.yml`
2. Install `lm_decoder` environment (environment of the [original repo](https://github.com/fwillett/speechBCI/)):
    - Create conda env: `conda create --name lm_decoder python=3.9 `
    - Activate env: `conda activate lm_decoder`
    - Install dependencies:
        - Run: `setup_lm_decoder_env.sh` 
        - OR:
        - Follow [installation guide](https://github.com/fwillett/speechBCI/tree/main?tab=readme-ov-file#installation) of original repo (also install LanguageModelDecoder as mentioned there).
3. Download and extract [dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)
3. Download and extract 3 gram and optionally 5 gram language models from [here](https://doi.org/10.5061/dryad.x69p8czpq). To execute decoding with the 5gram LM, you'll need ~750GB of RAM
4. Create `config.json` in root dir to specify variables that are valid across experiments:
```json
{
    "datasetPath": "[DATASET PATH]",
    "lm_3gram_dir": "[3 GRAM LM PATH]",
    "lm_5gram_dir": "[5 GRAM LM PATH]",
    "outputDir": "[OUTPUT BASE PATH]",
    "cacheDir": "[CACHE PATH TO STORE LARGE TEMP FILES AT]
}
```
## How to run

1. With either environment activated, convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. With `train_b2t` env active, train model (the args as used in the paper are automatically used). We used a compute node with a single A100 with 40GB+ of GPU memory and 32GB of RAM. A smaller GPU might also work. 
    - GRU (baseline): `python ./scripts/train_rnn.py`
    - Mamba: `python ./scripts/train_mamba.py`
    - MVTS Transformer: `python ./scripts/train_mvts.py`
    The final output directory of your training run will be printed at the end.
    You need it for the decoding described below
3. Run decoding using trained model (With `lm_decoder` env active)
    - 3 gram only decoding (< 100GB RAM needed): `python scripts/eval_competition_3gram.py --model_dir=[training run output dir]`
    - 3 gram + LLM decoding (< 100GB RAM needed): `python scripts/eval_competition_ngram_llm.py --model_dir=[training run output dir] --lm_variant=3gram`
    - 5 gram + LLM decoding (~750GB RAM needed): `python scripts/eval_competition_ngram_llm.py --model_dir=[training run output dir] --lm_variant=5gram`
    Note that when decoding Mamba model predictions, you might need to run any of the 3 decoding variants once within the `train_b2t` environment to generate the logits, as they can't be generated using the `lm_decoder` environment.
    The logits are then cached for that model output directory as a first step and are automatically loaded from disk when executed in the `lm_decoder` environment.
    Be aware that after the logits are generated and cached, the script executed within `train_b2t` env will fail since the lm decoding dependencies are not installed in that env. Just switch to `lm_decoder` then and execute the decoding scripts.



