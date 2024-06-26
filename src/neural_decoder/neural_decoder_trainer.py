import os
import pickle
import time
from uuid import uuid4

from edit_distance import SequenceMatcher
import hydra
from src.neural_decoder.b2p2t_model import PhonemeSampleBatch
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from src.neural_decoder.b2p2t_model import B2P2TModel, B2P2TModelArgsModel
from src.neural_decoder.mamba_model import MambaArgsModel, MambaModel
from src.neural_decoder.mvts_transformer_model import (
    MvtsTransformerModel,
    B2TMvtsTransformerArgsModel,
)
from .resnet import ResnetDecoder
from .gru_model import GRUDecoder
from .dataset import SpeechDataset


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: B2P2TModel):
        super().__init__()
        self.model = model

    def forward(self, neuralInput, dayIdx):
        batch = PhonemeSampleBatch(input=neuralInput, target=None)
        batch.day_idxs = dayIdx
        out = self.model.forward(batch)
        return out.logits


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(
        loadedData["train"],
        transform=None,
    )
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def trainModel(args):
    out_dir = os.path.join(args["outputDir"], str(uuid4()))
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(out_dir + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = load_model_based_on_args(args)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    if ("model" in args.keys()) and (args["model"] == "resnet"):
        optimizer = torch.optim.Adam(
            model.fc_decoder_out.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # if "model" in args.keys() and args["model"] == "resnet":
        #    # Reshapes input into (4,8,8) neural representation
        #    # splits = X.split(64, dim=-1)
        #    # channels = []
        #    # for split in splits:
        #    #     split = split.view(split.shape[0], split.shape[1], 8, 8)
        #    #     channels.append(split)
        #    # X = torch.stack(channels, dim=2)
        #
        #    # Reshapes input into (1, 16, 16) neural representation
        #    X = X.view(X.shape[0], X.shape[1], 1, 16, 16)

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - args["kernelLen"]) / args["strideLen"]).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - args["kernelLen"]) / args["strideLen"]).to(
                            torch.int32
                        ),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - args["kernelLen"]) / args["strideLen"]).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), out_dir + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(out_dir + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
    print("Saved results to", out_dir)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


def load_model_based_on_args(args):
    device = "cuda"
    model: torch.nn.Module
    if "model" in args.keys():
        if args["model"] == "mamba":
            mamba_config = MambaArgsModel(
                mamba_d_model=args["mamba_d_model"],
                mamba_n_layer=args["mamba_n_layer"],
                rms_norm=args["rms_norm"],
                residual_in_fp32=args["residual_in_fp32"],
                fused_add_norm=args["fused_add_norm"],
                feature_extractor_hidden_sizes=args["feature_extractor_hidden_sizes"],
                feature_extractor_activation=args["feature_extractor_activation"],
                classifier_hidden_sizes=args["classifier_hidden_sizes"],
                classifier_activation=args["classifier_activation"],
                input_dropout=args["dropout"],
            )
            mamba = MambaModel(
                config=mamba_config,
                vocab_size=args["nClasses"] + 1,
                in_size=args["nInputFeatures"] * args["kernelLen"],
            )
            b2p2t_model = B2P2TModel(
                B2P2TModelArgsModel(
                    gaussian_smooth_width=args["gaussianSmoothWidth"],
                    input_layer_nonlinearity="softsign",
                    unfolder_kernel_len=args["kernelLen"],
                    unfolder_stride_len=args["strideLen"],
                ),
                mamba,
            )
            if (
                "from_ourpipeline_checkpoint" in args
                and args["from_ourpipeline_checkpoint"] is not None
            ):
                b2p2t_model.load_state_dict(
                    torch.load(args["from_ourpipeline_checkpoint"])
                )

            model = ModelWrapper(b2p2t_model).to(device)
            if "from_checkpoint" in args and args["from_checkpoint"] is not None:
                model.load_state_dict(torch.load(args["from_checkpoint"]))
        elif args["model"] == "mvts":
            mvts_config = B2TMvtsTransformerArgsModel(
                dim_feedforward=args["dim_feedforward"],
                dropout=args["dropout"],
                num_layers=args["num_layers"],
                classifier_activation=args["classifier_activation"],
                num_heads=args["num_heads"],
                dim_model=args["dim_model"],
            )
            mvts = MvtsTransformerModel(
                config=mvts_config,
                vocab_size=args["nClasses"] + 1,
                in_size=args["nInputFeatures"] * args["kernelLen"],
            )
            b2p2t_model = B2P2TModel(
                B2P2TModelArgsModel(
                    gaussian_smooth_width=args["gaussianSmoothWidth"],
                    input_layer_nonlinearity="softsign",
                    unfolder_kernel_len=args["kernelLen"],
                    unfolder_stride_len=args["strideLen"],
                ),
                mvts,
            )
            if "from_checkpoint" in args and args["from_checkpoint"] is not None:
                b2p2t_model.load_state_dict(torch.load(args["from_checkpoint"]))
            model = ModelWrapper(b2p2t_model).to(device)
        else:
            model = ResnetDecoder(
                args["nClasses"] + 1,
                input_channels=1,
                neural_dim=args["nInputFeatures"],
                gaussianSmoothWidth=args["gaussianSmoothWidth"],
                nDays=24,
                dropout=args["dropout"],
            ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=24,
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
        ).to(device)
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)


if __name__ == "__main__":
    main()
