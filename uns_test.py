import copy
import os
from types import SimpleNamespace
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from dataset.kitti.parser import Parser
from modules.LifeHD import Model
from modules.LifeHD2 import LifeHD
from utils.eval_utils import eval_acc_multi_label, eval_nmi, eval_ri
from utils.plot_utils import plot_confusion_matrix, plot_novelty_detection, plot_tsne

from uns_extract import HistogramPool
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "life_hd_plots/end_high_hdc.pth"
BACKBONE_FILE = "extractor_model_0.4515.pth" 

transform = transforms.Compose([
    transforms.ToTensor()
])

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

# Replace with your dataset

test_loader = life_parser.get_valid_set()

print("\nLoading LifeHD model...")

opt_model = SimpleNamespace(
    dim=2000,
    hd_encoder="rp",
    max_classes=50,
    method="LifeHD",
    temperature = 0.01, # ...
)

opt_train = SimpleNamespace(
    dim=2000,
    epochs = ARCH["train"]["max_epochs"],
    warmup_batches = 1, # ????
    mask_mode = None, # or adaptive???
    rotation = 0.0,
    mask_dim = int(2000 * 0.6),
    beta = 3, # ???
    alpha = 0.3, # exponential smoothing factor for ...
    merge_mode = 'trim',# as long as its not no_trim?
    k_merge_min = 3,
    save_folder = "life_hd_plots",
    hit_th = 10,
)

model_temp = Model(opt_model, modelfile=BACKBONE_FILE, num_classes=28, device=DEVICE, patch_size=16).to(DEVICE)
model = LifeHD(opt_train, test_loader, test_loader, 28, model_temp, DEVICE, patch_size=16)

ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)

model.model.load_state_dict(ckpt["state_dict"])

model.model.classify_weights = ckpt["classify_weights"]
model.model.classify_sample_cnt = ckpt["classify_sample_cnt"]
model.model.dist_mean = ckpt["dist_mean"]
model.model.dist_std = ckpt["dist_std"]
model.model.cluster_labels = ckpt["cluster_labels"]
model.model.last_edit = ckpt["last_edit"]
model.model.cur_classes = ckpt["cur_classes"]

model.mask = ckpt["mask"]
model.cur_mask_dim = ckpt["cur_mask_dim"]

# model.eval()

pred_labels = []
true_labels = []
images = []

print("\nRunning inference on test set...")

with torch.no_grad():
    for idx, (image, _, label, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(test_loader, desc="Validating")):

        image = image.to(DEVICE)
        label = model.gen_label(label)
        label = label.to(DEVICE)

        outputs, hvs = model.model(image)
        preds = torch.argmax(outputs, dim=1)

        images.extend(image.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        true_labels.extend(label.cpu().numpy())

pred_labels = np.array(pred_labels)
true_labels = np.array(true_labels)
images = np.array(images)

print("\n==== Evaluation Results ====")

acc, cm = eval_acc_multi_label(true_labels, pred_labels, model.model.cluster_labels)
print("ACC:", acc)

nmi = eval_nmi(true_labels, pred_labels)
print("NMI:", nmi)

ri = eval_ri(true_labels, pred_labels)
print("RI:", ri)

print("\nSaving confusion matrix...")
plot_confusion_matrix(cm, test_loader, "stats_save")

print("Running t-SNE...")

plot_tsne(x=images, y_pred=pred_labels, y_true=true_labels, title="TSNE of features", fig_name="tsne.png")
print("\nDone! Results available in:", "stats_save")