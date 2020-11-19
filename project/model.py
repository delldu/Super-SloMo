"""Create model."""
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

import math
import os
import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from apex import amp
from tqdm import tqdm


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(
            outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        """ Forward. """
        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels,
                               3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(
            2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """ Forward . """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)),
                         negative_slope=0.1)
        return x


class UNet(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """ Forward."""
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    """

    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # create a grid
        # gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        # xxxx-????
        gridX, gridY = torch.meshgrid(torch.arange(W), torch.arange(H))

        self.W = W
        self.H = H
        # self.gridX = torch.tensor(
        #     gridX.t(), requires_grad=False, device=device)
        # self.gridY = torch.tensor(
        #     gridY.t(), requires_grad=False, device=device)

        self.gridX = gridX.t().to(device)
        self.gridY = gridY.t().to(device)
        # (Pdb) pp self.gridX.size()
        # torch.Size([512, 960])
        # (Pdb) pp self.gridX
        # tensor([[  0,   1,   2,  ..., 957, 958, 959],
        #         [  0,   1,   2,  ..., 957, 958, 959],
        #         [  0,   1,   2,  ..., 957, 958, 959],
        #         ...,
        #         [  0,   1,   2,  ..., 957, 958, 959],
        #         [  0,   1,   2,  ..., 957, 958, 959],
        #         [  0,   1,   2,  ..., 957, 958, 959]], device='cuda:0')

        # (Pdb) pp self.gridY.size()
        # torch.Size([512, 960])
        # (Pdb) pp self.gridY
        # tensor([[  0,   0,   0,  ...,   0,   0,   0],
        #         [  1,   1,   1,  ...,   1,   1,   1],
        #         [  2,   2,   2,  ...,   2,   2,   2],
        #         ...,
        #         [509, 509, 509,  ..., 509, 509, 509],
        #         [510, 510, 510,  ..., 510, 510, 510],
        #         [511, 511, 511,  ..., 511, 511, 511]], device='cuda:0')

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)
        Parameters
        ----------
            img : frame I1.
            flow : optical flow from I0 and I1: F_0_1.
        Returns
            frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        # (Pdb) flow.size()
        # torch.Size([1, 2, 512, 960])
        # pdb.set_trace()

        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)

        # pdb.set_trace()
        #  pp imgOut.size()
        # torch.Size([1, 3, 512, 960])

        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1.
t = torch.linspace(0.125, 0.875, 7)


def getFlowCoeff(indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : computation device (cpu/cuda). 
    Returns
            coefficients C00, C01, C10, C11.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C11 = C00 = - (1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))

    return \
        torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), \
        torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), \
        torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), \
        torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)


def getWarpCoeff(indices, device):
    """
    Gets coefficients used for calculating final intermediate 
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : computation device (cpu/cuda). 
    Returns
            coefficients C0 and C1.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), \
        torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)


def model_load(model, subname, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = state_dict["state_dict" + subname]
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)


def export_onnx_model():
    """Export onnx model."""

    import onnx
    from onnx import optimizer

    onnx_file = "output/video_slow.onnx"
    weight_file = "output/VideoSlow.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)

    input_names = ["input"]
    output_names = ["output"]
    # variable lenght axes
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('image_clean.onnx')"


def export_torch_model():
    """Export torch model."""

    script_file = "output/video_slow.pt"
    weight_file = "output/VideoSlow.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


def get_model(subname):
    """Create model."""

    model_setenv()
    # Flow Computer
    if subname == "FC":
        model = UNet(6, 4)
    elif subname == "AT":
        # arbitary - time
        model = UNet(20, 5)
    else:
        # backWarp, subname like 'backWarp: HxW'
        a = subname.split(':')[1].split('x')
        H, W = int(a[0]), int(a[1])
        model = backWarp(W, H, os.environ["DEVICE"])
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            # xxxx--modify here
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_device():
    """First call model_setenv. """
    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        os.environ["ENABLE_APEX"] = "YES"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def enable_amp(x):
    """Init Automatic Mixed Precision(AMP)."""
    if os.environ["ENABLE_APEX"] == "YES":
        x = amp.initialize(x, opt_level="O1")


def infer_perform():
    """Model infer performance ..."""

    model_setenv()

    model = VideoSlowModel()
    model.eval()
    device = model_device()
    model = model.to(device)
    enable_amp(model)

    progress_bar = tqdm(total=100)
    progress_bar.set_description("Test Inference Performance ...")

    for i in range(100):
        input = torch.randn(8, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        progress_bar.update(1)


if __name__ == '__main__':
    """Test model ..."""

    model = get_model()
    print(model)

    export_torch_model()
    export_onnx_model()

    infer_perform()
