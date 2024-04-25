from src.neural_decoder.neural_decoder_trainer import trainModel
import json
from src.neural_decoder.util import Config
import os

config = Config()
modelName = "mamba"

args = {}
args["outputDir"] = os.path.join(config.output_dir, modelName)
args["datasetPath"] = config.dataset_path
args["seqLen"] = 150
args["batchSize"] = 64
args["maxTimeSeriesLen"] = 1200
args["lrStart"] = 0.02
args["lrEnd"] = 0.01
args["nBatch"] = 10000  # 3000
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["whiteNoiseSD"] = 1.0
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["l2_decay"] = 1e-5
args["seed"] = 42

args["model"] = "mamba"

# Mamba args
args["feature_extractor_activation"] = "linear"
args["mamba_d_model"] = 256
args["mamba_n_layer"] = 64
args["weight_decay"] = 0.001
args["rms_norm"] = True  # True, gibt Fehler
args["residual_in_fp32"] = True
args["fused_add_norm"] = True  # True, gibt Fehler
args["classifier_hidden_sizes"] = []  # [256, 128, 64]
args["classifier_activation"] = "gelu"
args["feature_extractor_hidden_sizes"] = []  # [256, 256, 128, 64]

args["from_checkpoint"] = None

print("Training Mamba model with args: ")
print(json.dumps(args, indent=4))
trainModel(args)
