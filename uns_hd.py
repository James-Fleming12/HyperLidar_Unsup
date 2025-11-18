from types import SimpleNamespace
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from uns_extract import ContrastConv
from dataset.kitti.parser import Parser
from modules.Basic_HD import ExpHD_Dyn
from modules.LifeHD import Model as LifeHDModel
from modules.LifeHD2 import LifeHD
from modules.HDC_utils import Model_Dyn
from modules.ioueval import iouEval

NUM_CLASSES = 28
MODEL_DIR = "extractor_model_0.4515.pth"
PATCH_SIZE = 16

class DualHD:
    def __init__(self, ARCH, DATA, parser: Parser, life_parser: Parser, device="cuda", num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.feat_extract = ContrastConv(patch_size=PATCH_SIZE, num_classes=NUM_CLASSES)
        print(self.feat_extract.patch_size)
        print(self.feat_extract.conv.weight.size())

        checkpoint = torch.load(MODEL_DIR, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['state_dict'])
        else:
            self.feat_extract.load_state_dict(checkpoint)
        self.feat_extract.eval()

        self.train_data = parser.get_train_set()
        self.val_data = parser.get_valid_set()

        self.life_train = life_parser.get_train_set()
        self.life_val = life_parser.get_valid_set()

        self.epochs = 50 # temp for testing
        self.hd_dim = 2000
        self.randomness = 0.01 # ...
        self.validation_frequency = 5
        self.evaluator = iouEval(self.num_classes, device, [])

        opt_model = SimpleNamespace(
            dim=self.hd_dim,
            hd_encoder="rp",
            max_classes=50,
            method="LifeHD",
            temperature = self.randomness, # ...
        )

        opt_train = SimpleNamespace(
            dim=self.hd_dim,
            epochs = ARCH["train"]["max_epochs"],
            warmup_batches = 1, # ????
            mask_mode = None, # or adaptive???
            rotation = 0.0,
            mask_dim = int(self.hd_dim * 0.6),
            beta = 3, # ???
            alpha = 0.3, # exponential smoothing factor for ...
            merge_mode = 'trim',# as long as its not no_trim?
            k_merge_min = 3,
            save_folder = "life_hd_plots",
            hit_th = 10,
        )

        self.low_hd = Model_Dyn(ARCH, MODEL_DIR, "rp", self.hd_dim, 1, self.randomness, self.num_classes, self.device, patch_size=PATCH_SIZE)
        self.low_hd_trainer = ExpHD_Dyn(ARCH, DATA, self.low_hd, num_classes)

        self.high_hd = LifeHDModel(opt_model, MODEL_DIR, self.num_classes, self.device, patch_size=PATCH_SIZE)
        self.high_hd_trainer = LifeHD(opt_train, self.life_train, self.life_val, self.num_classes, self.high_hd, self.device, patch_size=PATCH_SIZE)

    def forward(self, x):
        pass

    def train(self):
        # self.train_low()
        self.train_high()

    def train_low(self):
        self.low_hd_trainer.train(self.train_data, self.low_hd) # initial training

        best_iou = 0.0
        for epoch in range(1, self.epochs+1):
            self.low_hd_trainer.retrain(self.train_data, self.low_hd, epoch)

            if epoch % self.validation_frequency == 0 or epoch == self.epochs:
                current_iou = self.low_hd_trainer.validate(self.val_data, self.low_hd, self.evaluator)
                if current_iou > best_iou:
                    best_iou = current_iou
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.low_hd.state_dict().copy(),
                        'best_iou': best_iou,
                        'classify_weights': self.low_hd.classify_weights.clone()
                    }, f'best_hdc_epoch_{epoch}_iou_{best_iou:.4f}.pth')

                    print(f"New model saved with IoU: {best_iou:.4f}")

    def train_high(self):
        self.high_hd_trainer.start()

def main():
    torch.cuda.empty_cache()
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

    parser = Parser(root = os.getcwd() + "/kitti_data/",
            train_sequences=[1,2,3,4,5,6,7,9,10],
            valid_sequences=[8],
            test_sequences=[11,12,13,14,15,16,17,18,19,20,21],
            labels=DATA["labels"],
            color_map=DATA["color_map"],
            learning_map=DATA["learning_map"],
            learning_map_inv=DATA["learning_map_inv"],
            sensor=ARCH["dataset"]["sensor"],
            max_points=ARCH["dataset"]["max_points"],
            batch_size=ARCH["train"]["batch_size"], # batch-size is 6 with current setup
            workers=ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False)
    life_parser = Parser(root = os.getcwd() + "/kitti_data/",
            train_sequences=[1,2,3,4,5,6,7,9,10],
            valid_sequences=[8],
            test_sequences=[11,12,13,14,15,16,17,18,19,20,21],
            labels=DATA["labels"],
            color_map=DATA["color_map"],
            learning_map=DATA["learning_map"],
            learning_map_inv=DATA["learning_map_inv"],
            sensor=ARCH["dataset"]["sensor"],
            max_points=ARCH["dataset"]["max_points"],
            batch_size = 1, # ...
            workers=ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False)

    net = DualHD(ARCH, DATA, parser, life_parser)
    net.train()

if __name__=="__main__":
    main()