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

import pdb

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoSlow.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/input", help="input video folder")
    parser.add_argument('--output', type=str, default="dataset/predict/output", help="output video folder")
    parser.add_argument("--scale", type=int, default=3, help='Increase the frames by Nx. Example scale=2 ==> 2x frames')

    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # Flow compute
    flow_compute = get_model("FC")
    model_load(flow_compute, "FC", args.checkpoint)
    flow_compute.to(device)
    flow_compute.eval()

    # Flow interpolate
    flow_interpolate = get_model("AT")
    model_load(flow_interpolate, "AT", args.checkpoint)
    flow_interpolate.to(device)
    flow_interpolate.eval()

    # if os.environ["ENABLE_APEX"] == "YES":
    #     from apex import amp
    #     model = amp.initialize(model, opt_level="O1")

    totensor = get_transform(train=False)
    toimage = reverse_transform()

    video = Video()
    video.reset(args.input)

    # Flow backwarp
    flow_backwarp = get_model("Backwarp: {:4d}x{:4d}".format(video.height, video.width))
    flow_backwarp.to(device)
    flow_backwarp.eval()

    progress_bar = tqdm(total=len(video))
    count = 1
    for index in range(len(video)):
        progress_bar.update(1)

        # videos format: TxCxHxW, Here T = 2
        frames = video[index]
        frame0 = frames[0].view(1, -1, video.height, video.width)
        frame1 = frames[1].view(1, -1, video.height, video.width)

        # Save orignal frame
        toimage(frame0.squeeze()).save("{}/{:06d}.png".format(args.output, count))
        count += 1

        I0 = frame0.to(device)
        I1 = frame1.to(device)        

        with torch.no_grad():
            flowOut = flow_compute(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]

        for j in range(1, args.scale):
            t = float(j) / args.scale
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            with torch.no_grad():
                g_I0_F_t_0 = flow_backwarp(I0, F_t_0)
                g_I1_F_t_1 = flow_backwarp(I1, F_t_1)

            with torch.no_grad():
                intrpOut = flow_interpolate( \
                torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, \
                g_I1_F_t_1, g_I0_F_t_0), \
                dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1

            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0

            with torch.no_grad():
                g_I0_F_t_0_f = flow_backwarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flow_backwarp(I1, F_t_1_f)

            wCoeff = [1 - t, t]

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + \
                wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            del g_I0_F_t_0_f, g_I1_F_t_1_f, F_t_0_f, F_t_1_f, F_t_0, F_t_1, intrpOut, V_t_0, V_t_1, wCoeff
            torch.cuda.empty_cache()

            # Save intermediate frame
            toimage(Ft_p[0].squeeze().cpu()).save("{}/{:06d}.png".format(args.output, count))
            count += 1

            del Ft_p
            torch.cuda.empty_cache()

        del F_0_1, F_1_0, flowOut, I0, I1, frame0, frame1
        torch.cuda.empty_cache()
