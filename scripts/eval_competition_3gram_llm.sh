#!/bin/bash
#SBATCH -A friedrich-compprogws2023
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --mail-user tobias.fiedler@student.hpi.de
#SBATCH --time=0-1:00:00
#SBATCH --partition=sorcery
#SBATCH  --constraint=ARCH:X86&GPU_SKU:A100


eval "$(conda shell.bash hook)"
conda activate lm_decoder
export PYTHONPATH="/hpi/fs00/home/tobias.fiedler/brain2text"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
srun python hpi_neural_seq_decoder/scripts/eval_competition_3gram_llm.py --model_dir=$1
