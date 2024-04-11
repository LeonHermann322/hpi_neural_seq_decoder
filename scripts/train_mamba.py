from hpi_neural_seq_decoder.src.neural_decoder.neural_decoder_trainer import trainModel


modelName = "mamba"

args = {}
args["outputDir"] = "/hpi/fs00/scratch/leon.hermann/b2t/speech_logs/" + modelName
args["datasetPath"] = "/hpi/fs00/scratch/leon.hermann/b2t/data/ptDecoder_ctc"
args["seqLen"] = 150
args["batchSize"] = 16
args["maxTimeSeriesLen"] = 1200
args["lrStart"] = 0.02
args["lrEnd"] = 0.02
args["nBatch"] = 10000  # 3000
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["whiteNoiseSD"] = 1
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["l2_decay"] = 1e-5
args["seed"] = 0

args["model"] = "mamba"

# Mamba args
args["feature_extractor_activation"] = "linear"
args["mamba_d_model"] = 1024
args["mamba_n_layer"] = 64
args["weight_decay"] = 0.001
args["rms_norm"] = False  # True, gibt Fehler
args["residual_in_fp32"] = True
args["fused_add_norm"] = False  # True, gibt Fehler
args["classifier_hidden_sizes"] = []  # [256, 128, 64]
args["classifier_activation"] = "gelu"
args["feature_extractor_hidden_sizes"] = []  # [256, 256, 128, 64]


trainModel(args)
