#!/usr/bin/env python3

import os
import argparse
# from pathlib import Path

# import yaml
import onnx
import torch
from torch.utils import model_zoo
# from addict import Dict as Adict

from retinaface.network import RetinaFace
from retinaface.pre_trained_models import models as RetinaModels


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c",
    #     "--config_path",
    #     type=Path,
    #     help="Path to the config.",
    #     required=True
    # )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name pretrained model or path to local file with model"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=True,
        help="The size of image that will send to input of model"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Name with path out onnx model"
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    """Load model
    Return PyTorch model
    """
    model = RetinaFace(
        name="Resnet50",
        pretrained=False,
        return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
        in_channels=256,
        out_channels=256,
    ).to(device)

    if os.path.exists(model_name):
        state_dict = torch.load(model_name)
    else:
        state_dict = model_zoo.load_url(
            RetinaModels[model_name].url, progress=True, map_location=device
        )
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    args = get_args()

    device = "cpu"

    model = load_model(args.model, device)
    model.eval()
    inputs = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    # Export the model
    torch.onnx.export(
        model,                     # model being run
        inputs,                    # model input (or a tuple for multiple inputs)
        args.out,                  # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        # dynamic_axes={             # variable lenght axes
        #     'input': {
        #         0: 'batch_size',
        #     },
        #     'output': {
        #         0: 'batch_size',
        #     }
        # }
    )

    # check
    onnx.checker.check_model(onnx.load(args.out))
