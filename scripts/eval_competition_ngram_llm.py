import re
import time
import pickle
import numpy as np
import torch

# from dataset import SpeechDataset
import os

# from nnDecoderModel import getDatasetLoaders

import pickle
import argparse
import json
from hpi_neural_seq_decoder.src.neural_decoder.dataset import SpeechDataset
from hpi_neural_seq_decoder.src.neural_decoder.neural_decoder_trainer import (
    getDatasetLoaders,
    load_model_based_on_args,
)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_dir", type=str, default=None, help="Path to model dir")
parser.add_argument(
    "--lm_variant",
    type=str,
    choices=["3gram", "5gram"],
    default=False,
    help="Use 5gram LM",
)
input_args = parser.parse_args()
print("Input args:", json.dumps(input_args.__dict__, indent=4))

with open(input_args.model_dir + "/args", "rb") as handle:
    args = pickle.load(handle)

print("Decoding for model with args", json.dumps(args, indent=4))

args["datasetPath"] = "/hpi/fs00/scratch/leon.hermann/b2t/data/ptDecoder_ctc"
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
    args["datasetPath"], args["batchSize"]
)
device = "cuda"

model = load_model_based_on_args(args)
model.load_state_dict(
    torch.load(os.path.join(input_args.model_dir, "modelWeights"), map_location=device)
)


model.eval()

rnn_outputs = {
    "logits": [],
    "logitLengths": [],
    "trueSeqs": [],
    "transcriptions": [],
}
partition = "competition"

# Caching rnn_outputs to allow for inference to be run in different environment than decoding
# If decoding does not work in inference env, simply execute once more with decoding env after caching rnn_outputs in inference env
if os.path.exists(input_args.model_dir + "/rnn_outputs.pkl"):
    print("Attempting to load rnn_outputs from cache")
    with open(input_args.model_dir + "/rnn_outputs.pkl", "rb") as f:
        rnn_outputs = pickle.load(f)
else:
    for i, testDayIdx in enumerate(
        [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    ):
        # for i, testDayIdx in enumerate(range(len(loadedData[partition]))):
        test_ds = SpeechDataset([loadedData[partition][i]])
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=0
        )
        for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                torch.tensor([testDayIdx], dtype=torch.int64).to(device),
            )
            pred = model.forward(X, dayIdx)
            adjustedLens = ((X_len - args["kernelLen"]) / args["strideLen"]).to(
                torch.int32
            )

            for iterIdx in range(pred.shape[0]):
                trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

                rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
                rnn_outputs["logitLengths"].append(
                    adjustedLens[iterIdx].cpu().detach().item()
                )
                rnn_outputs["trueSeqs"].append(trueSeq)

            transcript = loadedData[partition][i]["transcriptions"][j].strip()
            transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
            transcript = transcript.replace("--", "").lower()
            rnn_outputs["transcriptions"].append(transcript)

    with open(input_args.model_dir + "/rnn_outputs.pkl", "wb") as f:
        pickle.dump(rnn_outputs, f)

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

lmDir = (
    "/hpi/fs00/scratch/leon.hermann/languageModel"
    if input_args.lm_variant == "3gram"
    else "/hpi/fs00/scratch/leon.hermann/speech_5gram/lang_test"
)  # TODO: use correct 5gram path
ngramDecoder = lmDecoderUtils.build_lm_decoder(
    lmDir, acoustic_scale=0.5, nbest=100, beam=18
)


# LM decoding hyperparameters
acoustic_scale = 0.5
blank_penalty = np.log(7)
llm_weight = 0.5

llm_outputs = []
# Generate nbest outputs from 5gram LM
start_t = time.time()
decoded_transcripts = []

all_n_best = []
for j in range(len(rnn_outputs["logits"])):
    logits = rnn_outputs["logits"][j]
    logits = np.concatenate(
        [logits[:, 1:], logits[:, 0:1]], axis=-1
    )  # Blank is last token
    logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
    n_best = lmDecoderUtils.lm_decode(
        ngramDecoder,
        logits[0],
        blankPenalty=blank_penalty,
        returnNBest=True,
        rescore=input_args.lm_variant == "5gram",
    )
    all_n_best.append(n_best)
time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
print(f"3gram decoding took {time_per_sample} seconds per sample")


print("LLM rescoring")
print("Loading LLM model")
llm, llm_tokenizer = lmDecoderUtils.build_opt(
    cacheDir="/hpi/fs00/scratch/tobias.fiedler/brain2text",
    device="auto",
    load_in_8bit=True,
)
for i in range(len(rnn_outputs["transcriptions"])):
    new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
    rnn_outputs["transcriptions"][i] = np.array(new_trans)

start_t = time.time()
llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
    llm,
    llm_tokenizer,
    all_n_best[:],
    acoustic_scale,
    rnn_outputs,
    outputType="speech_sil",
    returnCI=True,
    lengthPenalty=0,
    alpha=llm_weight,
)
cer, cerIstart, cerIend = llm_out["cer"]
wer, werIstart, werIend = llm_out["wer"]
time_per_batch = (time.time() - start_t) / len(logits)
print(f"LLM decoding took {time_per_batch} seconds per sample")

out_file = input_args.model_dir + f"/{input_args.lm_variant}_llm_submission.txt"
with open(out_file, "w") as f:
    for transcript in llm_out["decoded_transcripts"]:
        f.write(transcript.strip() + "\n")

print("Saved submission txt in ", out_file)
