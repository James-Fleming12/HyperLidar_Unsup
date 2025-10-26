import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.network.ResNet import ResNet_34
from dataset.kitti.parser import Parser
from modules.Basic_HD import ExpHD

NUM_CLASSES = 28 # testing on SemanticKITTI

class HistogramPool(nn.Module):
    def __init__(self, patch_size, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch_size, channels, height, width = x.shape
        
        if channels > 1:
            x = x[:, 0:1, :, :]
            
        h_out = height // self.patch_size
        w_out = width // self.patch_size
        num_patches = h_out * w_out
        patch_area = self.patch_size * self.patch_size
        
        patches = F.unfold(x.float(), kernel_size=self.patch_size, stride=self.patch_size)

        patches = patches.view(batch_size, channels, patch_area, num_patches).long()
        valid_mask = (patches != 0)

        patches_clean = patches.clone()
        patches_clean[~valid_mask] = 0

        patches_flat = patches_clean.view(batch_size, channels * patch_area, num_patches)
        patches_one_hot = F.one_hot(patches_flat.permute(0, 2, 1), num_classes=self.num_classes)

        patches_one_hot[~valid_mask.view(batch_size, patch_area * channels, num_patches).permute(0, 2, 1)] = 0
        histogram = patches_one_hot.sum(dim=2)
        
        output = histogram.permute(0, 2, 1).contiguous()
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

        self.criterion = nn.NLLLoss(ignore_index=0)

        self.scale = 0.05
        self.pre_epochs = 10

    def forward(self, x):
        """
        returns low and high level features
        """
        low = self.net(x, only_feat = True)
        high = self.conv(low)
        return low, high
    
    def check_gradients(self):
        """Check gradient flow through the network"""
        print(f"\n{'='*50}")
        print("GRADIENT DIAGNOSTICS")
        print(f"{'='*50}")
        
        total_norm = 0
        has_vanish = False
        has_explode = False
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.abs().mean().item()
                total_norm += grad_norm ** 2
                
                status = "OK"
                if grad_mean < 1e-7:
                    status = "VANISHING"
                    has_vanish = True
                elif grad_mean > 1.0:
                    status = "EXPLODING"
                    has_explode = True
                
                print(f"{name:40} | Norm: {grad_norm:8.6f} | Mean: {grad_mean:8.6f} | {status}")
            else:
                print(f"{name:40} | No gradients")
        
        total_norm = total_norm ** 0.5
        print(f"\nTotal Gradient Norm: {total_norm:.6f}")
        print(f"Gradient Status: {'VANISHING' if has_vanish else 'EXPLODING' if has_explode else 'HEALTHY'}")
        print(f"{'='*50}\n")
        
        return total_norm

    def loss(self, low, high, label, current_epoch):
        low_class = self.low_classifier(low)
        low_class = F.log_softmax(low_class, dim=1)
        low_loss = self.criterion(low_class, label.long())

        if current_epoch < self.pre_epochs:
            return low_loss

        high_class = self.high_classifier(high)
        high_class = F.log_softmax(high_class, dim=1)

        high_label = self.pool(label)

        high_sums = high_label.sum(dim=1, keepdim=True)
        high_sums = torch.where(high_sums == 0, torch.ones_like(high_sums), high_sums) # avoid division by 0
        # high_label = high_label / high_sums # normalize so loss terms are not imbalanced

        # low_loss = -label_onehot * low_class
        # low_loss = low_loss.sum(dim=1)
        # low_loss = (low_loss * mask.float()).mean() # only average over valid pixels
        valid_high_patches = (high_label.sum(dim=1) > 0).float()
        num_valid_patches = valid_high_patches.sum()
        
        if num_valid_patches > 0:
            high_loss = (-high_label * high_class).sum(dim=1)
            high_loss = (high_loss * valid_high_patches).sum() / num_valid_patches
        else:
            high_loss = torch.tensor(0.0).to(high_class.device)

        total_loss = low_loss + self.scale * high_loss

        return total_loss

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

    # test_batch = parser.get_train_batch() # first are inputs, third are labels?
    # test_in = test_batch[0][0] 
    # test_in = test_in[None, ...] # 1, 5, 64, 512
    # test_la = test_batch[1][0]
    # test_la = test_la[None, ...] # 1, 64, 512
    net = TestContrastConv()

    # low, high = net(test_in) # low: 1, 128, 64, 512   high: 1, 128, 4, 32

    # net.loss(low, high, test_la)
    train_dataset = parser.get_train_set()
    val_dataset = parser.get_valid_set()
    # optimizer = torch.optim.Adam(net.parameters(), lr=ARCH["train"]["decay"]["lr"])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # changes from 0.000001
    best_val_loss = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    for epoch in range(ARCH["train"]["max_epochs"]):
        net.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, curr in enumerate(train_dataset):
            curr_in = curr[0].to(device)
            curr_label =  curr[2].to(device)
            
            optimizer.zero_grad()
            low, high = net(curr_in)
            loss = net.loss(low, high, curr_label, epoch)
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

            # if batch_idx % 1000 == 0:
            #     net.check_gradients()
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for curr in val_dataset:
                curr_in = curr[0].to(device)
                curr_label = curr[2].to(device)
                low, high = net(curr_in)
                loss = net.loss(low, high, curr_label, epoch)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        scheduler.step(avg_val_loss)
        
        print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}")

        if avg_val_loss < best_val_loss:
            if epoch > net.pre_epochs:
                best_val_loss = avg_val_loss
            torch.save(net.state_dict(), 'extractor_model.pth')
            print(f"Model saved in extractor_model.pth with loss of {best_val_loss:.4f}")

if __name__ == "__main__":
    main()