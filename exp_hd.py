import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.network.ResNet import ResNet_34
from dataset.kitti.parser import Parser
from modules.Basic_HD import ExpHD

NUM_CLASSES = 28
MODEL_DIR = "extractor_model.pth"

class DualHD:
    def __init__(self, device="cuda", num_classes=NUM_CLASSES):
        super().__init__()
        self.num_class = num_classes
        self.feat_extract = ResNet_34(self.num_class, False)

        checkpoint = torch.load(MODEL_DIR, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['state_dict'])
        else:
            self.feat_extract.load_state_dict(checkpoint)
        self.feat_extract.eval()
        

    def forward(self, x):
        pass

def main():
    net = DualHD()

if __name__=="__main__":
    main()