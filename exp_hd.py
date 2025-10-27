import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_extract import ContrastConv
from dataset.kitti.parser import Parser
from modules.Basic_HD import ExpHD_Dyn
from modules.LifeHD import LifeHD
from modules.HDC_utils import Model_Dyn
from modules.ioueval import iouEval

NUM_CLASSES = 28
MODEL_DIR = "extractor_model.pth"

class DualHD:
    def __init__(self, ARCH, DATA, device="cuda", num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.feat_extract = ContrastConv(patch_size=4, num_classes=NUM_CLASSES)

        checkpoint = torch.load(MODEL_DIR, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['state_dict'])
        else:
            self.feat_extract.load_state_dict(checkpoint)
        self.feat_extract.eval()

        self.low_hd = Model_Dyn(ARCH, MODEL_DIR, "rp", 1, 0.5, self.num_classes, 'cuda')
        self.low_hd_trainer = ExpHD_Dyn(ARCH, DATA, self.low_hd, num_classes)

        self.epochs = 2 # temp for testing
        self.evaluator = iouEval(self.num_classes, device, [])

    def forward(self, x):
        pass

    def train(self, parser: Parser):
        trainer = parser.get_train_set()
        val_train = parser.get_valid_set()
        print("PRETRAINING STARTING")
        self.low_hd_trainer.train(trainer, self.low_hd) # initial training

        print("TRAINING STARTING")
        self.low_hd_trainer.retrain(trainer, self.low_hd, 1)

        print("VALIDATION TRAINING STARTING")
        self.low_hd_trainer.validate(val_train, self.low_hd, self.evaluator)

        # for epoch in range(1, self.epochs+1):
        #     self.low_hd_trainer.retrain(trainer, self.low_hd, epoch)

        #     if epoch % self.validation_frequency == 0 or epoch == self.epochs:
        #         self.low_hd_trainer.validate(val_train, self.low_hd, self.evaluator)

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

    net = DualHD(ARCH, DATA)
    net.train(parser)

if __name__=="__main__":
    main()