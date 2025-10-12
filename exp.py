import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.network.ResNet import ResNet_34
from dataset.kitti.parser import Parser

NUM_CLASSES = 28 # testing on SemanticKITTI

class TestContrastConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ResNet_34(NUM_CLASSES, False)
        self.patch_size = 16 # works since input is image of 512 * 64
        self.conv = nn.Conv2d(28, 28, self.patch_size, stride=self.patch_size, padding=0)

    def forward(self, x):
        low = self.net(x)
        high = self.conv(low)
        return low, high

    def loss(self):
        pass

class Tester(nn.Module):
    def __init__(self):
        self.enc = TestContrastConv()

    def forward(self, x):
        pass

def main():
    try:        # open arch config file
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

    train_in = parser.get_train_batch()[0][0]
    train_in = train_in[None, ...]
    print(train_in.size())
    print(train_in.size())
    testnet = TestContrastConv()

    temp1, temp2 = testnet(train_in)
    print(temp1.size())
    print(temp2.size())

if __name__ == "__main__":
    main()