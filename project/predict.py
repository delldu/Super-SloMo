"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:54:18 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import get_model, model_load, model_setenv
from data import Video, get_transform, reverse_transform

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoSlow.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/input", help="input video folder")
    parser.add_argument('--output', type=str, default="dataset/predict/output", help="output video folder")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model("FC")
    model_load(model, "FC", args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = get_transform(train=False)
    toimage = reverse_transform()

    video = Video()
    video.reset(args.input)
    progress_bar = tqdm(total=len(video))

    for index in range(len(video)):
        progress_bar.update(1)

        print(video[index].size())

        # image = Image.open(filename).convert("RGB")
        # input_tensor = totensor(image).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()

        # toimage(output_tensor.cpu()).save(
        #     "{}/{:06d}.png".format(args.output, index + 1))
