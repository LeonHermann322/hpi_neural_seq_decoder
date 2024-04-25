from src.neural_decoder.neural_decoder_trainer import trainModel
from src.neural_decoder.util import Config
import os

config = Config()

modelName = "resnet"

args = {}
args["outputDir"] = os.path.join(config.output_dir, modelName)
args["datasetPath"] = config.dataset_path
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 8
args["lrStart"] = 0.02
args["lrEnd"] = 0.02
args["nUnits"] = 1024
args["nBatch"] = 10000  # 3000
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True
args["l2_decay"] = 1e-5

args["model"] = "resnet"


trainModel(args)
