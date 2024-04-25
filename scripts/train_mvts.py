from hpi_neural_seq_decoder.src.neural_decoder.neural_decoder_trainer import trainModel
import json

modelName = "mvts"

args = {}
args["outputDir"] = (
    "/hpi/fs00/scratch/tobias.fiedler/brain2text/speech_logs/" + modelName
)
args["datasetPath"] = "/hpi/fs00/scratch/leon.hermann/b2t/data/ptDecoder_ctc"
args["seqLen"] = 150
args["batchSize"] = 64
args["maxTimeSeriesLen"] = 1200
args["lrStart"] = 0.005
args["lrEnd"] = 0.005
args["nBatch"] = 10000  # 3000
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.3
args["whiteNoiseSD"] = 1.0
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["l2_decay"] = 1e-5
args["seed"] = 42

args["model"] = "mvts"

# Mvts args
args["dim_feedforward"] = 256
args["num_layers"] = 6
args["num_heads"] = 2
args["classifier_activation"] = "gelu"
args["dim_model"] = 1024

args["from_checkpoint"] = None

print("Training Mvts model with args: ")
print(json.dumps(args, indent=4))
trainModel(args)
