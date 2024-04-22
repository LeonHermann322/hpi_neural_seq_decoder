from hpi_neural_seq_decoder.src.neural_decoder.neural_decoder_trainer import trainModel
import json

modelName = "mamba"

args = {}
args["outputDir"] = (
    "/hpi/fs00/scratch/tobias.fiedler/brain2text/speech_logs/" + modelName
)
args["datasetPath"] = "/hpi/fs00/scratch/leon.hermann/b2t/data/ptDecoder_ctc"
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

args["from_ourpipeline_checkpoint"] = (
    "/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_mamba/2024-03-26_10#39#22/model.pt"
)

print("Training Mamba model with args: ")
print(json.dumps(args, indent=4))
trainModel(args)
