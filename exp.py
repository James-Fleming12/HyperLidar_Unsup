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
    
class HistogramPool(nn.Module):
    def __init__(self, patch_size, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3: # mainly for testing to add batch dimension
            x = x.unsqueeze(0)
        batch_size, channels, height, width = x.shape
        h_out = height // self.patch_size
        w_out = width // self.patch_size
        patches = F.unfold(x.float(), kernel_size=self.patch_size, stride=self.patch_size)

        patches = patches.view(batch_size, channels, self.patch_size * self.patch_size, -1).long()
        patches = patches - 1
        # num_patches = patches.shape[-1]

        patches_one_hot = torch.zeros(batch_size, self.num_classes, self.patch_size * self.patch_size, h_out * w_out, device=x.device)
        
        patches_expanded = patches.expand(-1, self.num_classes, -1, -1)
        class_indices = torch.arange(self.num_classes, device=x.device).view(1, -1, 1, 1)
        mask = (patches_expanded == class_indices)
        patches_one_hot[mask] = 1
        
        output = patches_one_hot.sum(dim=2)
        output = output.view(batch_size, self.num_classes, h_out, w_out)
        
        return output

class TestContrastConv(nn.Module):
    def __init__(self, patch_size=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = ResNet_34(num_classes, False)
        self.patch_size = patch_size # 16 works since input is image of 512 * 64
        self.conv = nn.Conv2d(128, 128, self.patch_size, stride=self.patch_size, padding=0)

        self.low_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.pool = HistogramPool(self.patch_size, num_classes)
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.scale = 0.5 # relatively low rn cause the high-level loss is really high

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
        low_class = F.log_softmax(low_class, dim=1)
        high_class = F.log_softmax(high_class, dim=1)

        high_label = self.pool(label)
        label_onehot = F.one_hot(label.long(), num_classes=self.num_classes)
        label_onehot = label_onehot.permute(0, 3, 1, 2)

        low_loss = -label_onehot * low_class
        low_loss = low_loss.sum(dim=1).mean()

        high_loss = -high_label * high_class
        high_loss = high_loss.sum(dim=1).mean()

        return low_loss + self.scale * high_loss

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

    # test_test_la = torch.randint(low=1, high=29, size=(1, 64, 512))

    low, high = testnet(test_in) # low: 1, 128, 64, 512   high: 1, 128, 4, 32

    testnet.loss(low, high, test_la)

if __name__ == "__main__":
    main()