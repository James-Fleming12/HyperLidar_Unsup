import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.network.ResNet import ResNet_34
from dataset.kitti.parser import Parser
from modules.Basic_HD import ExpHD

NUM_CLASSES = 28 # testing on SemanticKITTI

class ModePool2D(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.unsqueeze(1) # unfold needs a 4D tensor
        x = x.float() # also needs it to be a float tensor
        batch_size, channels, height, width = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)

        patch_area = self.patch_size * self.patch_size
        patches = patches.view(batch_size, channels, patch_area, -1)
        patches = patches.to(orig_dtype)
        mode_values, _ = torch.mode(patches, dim=2)

        output = mode_values.view(batch_size, height // self.patch_size, width // self.patch_size)

        return output

class TestContrastConv(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()
        self.net = ResNet_34(NUM_CLASSES, False)
        self.patch_size = patch_size # 16 works since input is image of 512 * 64
        self.conv = nn.Conv2d(128, 128, self.patch_size, stride=self.patch_size, padding=0)

        self.low_classifier = nn.Conv2d(128, NUM_CLASSES, kernel_size=1)
        self.high_classifier = nn.Conv2d(128, NUM_CLASSES, kernel_size=1)
        self.modepool = ModePool2D(self.patch_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        returns low and high level features
        """
        low = self.net(x, only_feat = True)
        high = self.conv(low)
        return low, high

    def loss(self, low, high, label):
        low_class = self.low_classifier(low)
        high_class = self.high_classifier(high)

        high_label = self.modepool(label) # has way too many zeroes and not even close to enough ones, might be a problem
                                          # might need to make a function that gives a one hot vector of all the values within the patch

        ones = torch.sum(high_label == 1).item()
        zeros = torch.sum(high_label == 0).item()
        print(f"Ones: {ones}, Zeroes: {zeros}")


class Tester(nn.Module):
    def __init__(self):
        self.enc = TestContrastConv()

    def forward(self, x):
        pass

def main():
    try: # open arch config file
        ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:    # open data config file
        DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()
    parser = Parser(root = os.getcwd() + "/nuscenes_kitti/",
            train_sequences=[0,1],
            valid_sequences=[0,1],
            test_sequences=[0,1],
            labels=DATA["labels"],
            color_map=DATA["color_map"],
            learning_map=DATA["learning_map"],
            learning_map_inv=DATA["learning_map_inv"],
            sensor=ARCH["dataset"]["sensor"],
            max_points=ARCH["dataset"]["max_points"],
            batch_size=ARCH["train"]["batch_size"],
            workers=ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False)

    test_batch = parser.get_train_batch() # first are inputs, second are labels
    test_in = test_batch[0][0] 
    test_in = test_in[None, ...] # 1, 5, 64, 512
    test_la = test_batch[1][0]
    test_la = test_la[None, ...] # 1, 64, 512
    testnet = TestContrastConv()

    low, high = testnet(test_in) # low: 1, 128, 64, 512   high: 1, 128, 4, 32

    testnet.loss(low, high, test_la)

if __name__ == "__main__":
    main()