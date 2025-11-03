import os
import copy
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from tqdm import tqdm

from exp_extract import HistogramPool
from modules.LifeHD import Model
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_confusion_matrix, plot_tsne

novelty_detect = []
class_shift = []
VAL_CNT = 10

EIGEN_MAX = 2000 # just to make testing a little faster

def get_nc_laplacian(class_hvs, batch_idx, opt):
    """
    Obtain the number of clusters by searching for plateaus
    in the sorted eigenvalues

    Args:
        class_hvs: extracted class hypervectors by calling model.extract_class_hv()
        batch_idx: the batch index in the training stream for logging
        opt: arguments

    Returns:
        nc: the number of clusters as the start of the plateau
        L: the k neighbors graph of the input class_hvs
        U: the eigenvectors of L
    """
    G = kneighbors_graph(class_hvs, 3, include_self=True).toarray()
    L = csgraph.laplacian(G)
    # print(L)

    (S, U) = np.linalg.eig(L)
    S, U = np.real(S), np.real(U)
    ixs = np.argsort(S)  # Sort, ascending
    S, U = S[ixs], U[:, ixs]
    U = U[:, S > 0]
    S = S[S > 0]
    S = S / S.max()

    if batch_idx == opt.warmup_batches or batch_idx % 50 == 0:
        # Plot sorted eigenvalues
        fig = plt.figure()
        plt.plot(np.arange(S.size), S)
        plt.title('Idx: {}'.format(
            batch_idx
        ))
        plt.savefig(os.path.join(opt.save_folder, 'eigenvalue_{}.png'.format(batch_idx)))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(np.arange(U.shape[0]), np.sort(U[:, 1]))
        plt.title('Idx: {}'.format(
            batch_idx
        ))
        plt.savefig(os.path.join(opt.save_folder, 'fiedlervector_{}.png'.format(batch_idx)))
        plt.close(fig)

    return -1, L, U  # Haven't figure out how to get nc



def get_nc(class_hvs, pair_simil, thres, batch_idx, opt, warmup_done):
    """
    Obtain the number of clusters by searching for plateaus
    in the sorted eigenvalues

    Args:
        class_hvs: extracted class hypervectors by calling model.extract_class_hv()
        pair_simil: pairwise similarity between class hypervectors
        thres: the threshold for the pairwise similarity neighborhood
        batch_idx: the batch index in the training stream for logging
        opt: arguments
        warmup_done: whether warmup has been done

    Returns:
        nc: the number of clusters as the start of the plateau
        L: the k neighbors graph of the input class_hvs
        U: the eigenvectors of L
    """
    print('warmup done:', warmup_done)
    #if not warmup_done:
    L = kneighbors_graph(class_hvs, 4, include_self=True).toarray()
    #else:
    #    print('not warmup!!')
    #    L = (pair_simil > thres).astype('int')

    # Compute the eigenvalues and eigenvectors of L
    (S, U) = np.linalg.eig(L)
    S, U = np.real(S), np.real(U)
    ixs = np.argsort(-1 * S)  # Sort, descending
    S, U = S[ixs], U[:, ixs]
    U = U[:, S > 0]
    S = S[S > 0]
    S = S / S.max()

    nc = np.argmax(S < 0.1)

    print('Idx: {} nc={}'.format(
        batch_idx, nc
    ))

    return nc, L, U

class LifeHD():
    def __init__(self, opt, train_loader, val_loader, num_classes, model: Model, device):
        self.opt = opt
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        self.model = model.to(self.device)

        self.warmup_done = False

        self.mask = torch.ones(opt.dim, device = self.device).type(torch.bool)
        self.cur_mask_dim = self.opt.dim
        self.last_novel = 0

        self.trim = 0 # trim and merge stats
        self.merge = 0

        self.patch_size = 4
        self.pool = HistogramPool(self.patch_size, self.num_classes)

    def start(self):
        for epoch in range(1, self.opt.epochs+1):
            # train for one epoch
            time1 = time.time() 
            self.train(epoch)

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # final validation
            acc = self.validate(epoch, len(self.train_loader), True, 'final')
            print('Stream final acc: {}'.format(acc))

    def warmup(self, idx, sample_hv, label):
        if idx == 0:
            self.warmup_hvs = sample_hv
            self.warmup_labels = label

        elif idx < self.opt.warmup_batches:
            self.warmup_hvs = torch.cat((self.warmup_hvs, sample_hv), dim=0)
            self.warmup_labels = torch.cat((self.warmup_labels, label))

        elif idx >= self.opt.warmup_batches:

            if self.warmup_hvs.shape[0] > EIGEN_MAX:
                print(f"Subsampled from {self.warmup_hvs.shape[0]} to {EIGEN_MAX} for eigen decomposition")
                indices = np.random.choice(self.warmup_hvs.shape[0], EIGEN_MAX, replace=False)
                self.warmup_hvs = self.warmup_hvs[indices]
                self.warmup_labels = self.warmup_labels[indices]

            nc, L, U = get_nc(self.warmup_hvs.cpu().numpy(), None, None, idx, self.opt, self.warmup_done)
            nc = nc if 0 < nc < self.model.max_classes else int(0.5 * self.model.max_classes)

            K2 = KMeans(nc)
            K2.fit(U[:, :nc])

            # init clusters 
            for ix in range(nc):
                cluster_mask = (K2.labels_ == ix)
                self.model.classify_weights[ix] = self.warmup_hvs[cluster_mask].sum(dim=0)  # size 1xD
                self.model.classify_sample_cnt[ix] = cluster_mask.sum()

                labels_in_cluster = self.warmup_labels[cluster_mask].tolist()
                self.model.cluster_labels[ix] = self.gen_store_labels(labels_in_cluster)

                # Only update mean and std when there are more than 1 sample in the init cluster
                #if cluster_mask.sum() > 1:
                dist_to_cen = F.normalize(self.warmup_hvs[cluster_mask]) @ \
                    F.normalize(self.model.classify_weights[ix].view(1, -1)).T  
                    # Should be size sample_cntx1
                self.model.dist_mean[ix] = torch.mean(dist_to_cen)  # scalar
                self.model.dist_std[ix] = torch.mean(torch.abs(dist_to_cen - self.model.dist_mean[ix]))  # scalar

                self.model.last_edit[ix] = idx

            self.model.cur_classes = nc

            weight_sum = torch.abs(self.model.classify_weights[:nc].sum(dim=0))
            sort_idx = torch.argsort(weight_sum, descending=True)
            self.mask = torch.zeros(self.opt.dim, device=self.device).type(torch.bool)
            self.mask[sort_idx[:self.opt.mask_dim]] = 1
            self.cur_mask_dim = self.opt.mask_dim

            print('init # of clusters after warmup: {}'.format(nc))

            del self.warmup_hvs
            del self.warmup_labels
            self.warmup_done = True

    def train(self, epoch):
        val_freq = np.floor(len(self.train_loader) / VAL_CNT).astype('int')
        batchs_per_class = np.floor(len(self.train_loader) / self.num_classes).astype('int')

        with torch.no_grad():
            for idx, (image, _, label, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(self.train_loader, desc="Training")):
                image = image.to(self.device)
                label = self.gen_label(label)
                label = label.to(self.device)
                outputs, sample_hv = self.model(image, self.mask)

                # validation
                if idx > self.opt.warmup_batches and idx % val_freq == 0:
                    if idx > self.opt.warmup_batches + 1 and self.opt.merge_mode  != 'no_trim':
                        self.trim_clusters()

                    if self.opt.merge_mode  != 'no_merge':
                        pair_simil, class_hvs = self.model.extract_pair_simil(self.mask)
                        thres = self.model.dist_mean[:self.model.cur_classes].mean().cpu
                        nc, _, U = get_nc(class_hvs, pair_simil, thres, idx, self.opt, self.warmup_done)

                        if self.opt.k_merge_min < nc < self.model.max_classes:
                            self.merge_clusters(U, nc, class_hvs, idx)

                    # acc, purity = self.validate(epoch, idx+1, False, 'after')
                    # print('Validate stream: [{}][{}/{}]\tacc: {} purity: {}'.format(epoch, idx + 1, len(self.train_loader), acc, purity))
                    # sys.stdout.flush()

                if self.opt.mask_mode == 'adaptive' and idx - self.last_novel > 3:
                    weight_sum = torch.abs(self.model.classify_weights[:self.model.cur_classes].sum(dim=0))
                    sort_idx = torch.argsort(weight_sum, descending=True)
                    self.mask = torch.zeros(self.opt.dim, device=self.device).type(torch.bool)
                    self.mask[sort_idx[:self.opt.mask_dim]] = 1
                    self.cur_mask_dim = self.opt.mask_dim

                if not self.warmup_done:
                    self.warmup(idx, sample_hv, label)
                else:
                    # normal session after warmup
                    simil_to_class, pred_class = torch.max(outputs, dim=-1)
                    pred_class_samples = self.model.classify_sample_cnt[pred_class]

                    assert pred_class_samples.min() > 0, 'Predicted class {} has zero samples!' # ...

                    simil_threshold = self.model.dist_mean[pred_class] - self.opt.beta * self.model.dist_std[pred_class]

                    novel_detect_mask = (simil_to_class < simil_threshold) & (pred_class_samples > 10)

                    self.add_sample_hv_to_exist_class(sample_hv[~novel_detect_mask], 
                                                      pred_class[~novel_detect_mask], 
                                                      simil_to_class[~novel_detect_mask], 
                                                      idx, label[~novel_detect_mask])


    def validate(self, epoch, loader_idx, plot, mode):
        test_samples, test_embeddings = None, None
        pred_labels, test_labels = [], []

        with torch.no_grad():
            for idx, (image, _, label, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(self.train_loader, desc="Testing")):
                image = image.to(self.device)
                label = self.gen_label(label)
                label = label.to(self.device)
                
                outputs, _ = self.model(image, self.mask)
                predictions = torch.argmax(outputs, dim=-1)

                pred_labels += predictions.detach().cpu().tolist()
                test_labels += label.cpu().tolist()

                embeddings = self.model.encode(image).detach().cpu().numpy()
                test_bsz = image.size(0)
                if test_embeddings is None:
                    test_samples = image.squeeze().view((test_bsz, -1)).cpu().numpy()
                    test_embeddings = embeddings
                else:
                    test_samples = np.concatenate((test_samples, image.squeeze().view((test_bsz, -1)).cpu().numpy()),axis=0)
                    test_embeddings = np.concatenate((test_embeddings, embeddings), axis=0)

                pred_labels = np.array(pred_labels).astype(int)
                print(np.unique(pred_labels))
                test_labels = np.array(test_labels).astype(int)
                acc, purity, cm = eval_acc(test_labels, pred_labels)
                print('Acc: {}, purity: {}'.format(acc, purity))

                nmi = eval_nmi(test_labels, pred_labels)
                print('NMI: {}'.format(nmi))

                ri = eval_ri(test_labels, pred_labels)
                print('RI: {}'.format(ri))

                with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
                    f.write('{epoch},{idx},{acc},{purity},{nmi},{ri},{nc},{trim},{merge}\n'.format(
                        epoch=epoch, idx=loader_idx, acc=acc, purity=purity,
                        nmi=nmi, ri=ri, nc=self.model.cur_classes, 
                        trim=self.trim, merge=self.merge
                    ))

                if plot:
                    # plot the tSNE of raw samples with predicted labels
                    plot_tsne(test_embeddings, np.array(pred_labels), np.array(test_labels),
                            title='embeddings {} {} {} {}'.format(self.opt.method, self.opt.dataset, acc, mode),
                            fig_name=os.path.join(self.opt.save_folder,
                                                    '{}_emb_{}_{}_{}.png'.format(
                                                        loader_idx, self.opt.method, self.opt.dataset, mode)))
                    
                    np.save(os.path.join(self.opt.save_folder, 'confusion_mat'), cm)
                    # plot confusion matrix
                    plot_confusion_matrix(cm, self.opt.dataset, self.opt.save_folder)
        
            return acc, purity

    def add_sample_hv_to_exist_class(self, sample_hv, pred_class, simil_to_class, batch_idx, label):
        pred_class_set = np.unique(pred_class.cpu().numpy())

        for cs in pred_class_set:
            mask = (pred_class == cs)
            old_sample_hv = copy.deepcopy(self.model.classify_weights[cs].view(1, -1))
            old_sample_cnt = self.model.classify_sample_cnt[cs].item()
            self.model.classify_weights[cs, self.mask] += sample_hv[mask][:, self.mask].sum(dim=0)
            self.model.classify_sample_cnt[cs] += mask.sum()
            
            self.add_cluster_label(cs, label)

            if old_sample_cnt > 1:
                self.model.dist_std[cs] = \
                    self.opt.alpha * torch.mean(torch.abs(simil_to_class[mask] - self.model.dist_mean[cs])) + \
                    (1 - self.opt.alpha) * self.model.dist_std[cs]
                self.model.dist_mean[cs] = \
                    self.opt.alpha * simil_to_class[mask].mean() + \
                    (1 - self.opt.alpha) * self.model.dist_mean[cs]

            else:
                dist_to_cen = \
                    F.normalize(torch.cat((old_sample_hv[:, self.mask], sample_hv[mask][:, self.mask]), dim=0)) @ \
                    F.normalize(self.model.classify_weights[cs, self.mask].view(1, -1)).T
                self.model.dist_mean[cs] = torch.mean(dist_to_cen)  # scalar
                self.model.dist_std[cs] = torch.mean(
                    torch.abs(dist_to_cen - self.model.dist_mean[cs]))  # scalar

            self.model.last_edit[cs] = batch_idx

    def add_sample_hv_to_novel_class(self, sample_hv, batch_idx):
        pass

    def add_cluster_label(self, cluster, label):
        if label in self.model.cluster_labels[cluster]:
            self.model.cluster_labels[cluster][label] += 1
        else:
            self.model.cluster_labels[cluster][label] = 1

    def gen_label(self, label):
        label = self.pool(label)
        return torch.argmax(label, dim=1).flatten()

    def gen_store_labels(self, labels_in_cluster):
        from collections import Counter
        label_counts = Counter(labels_in_cluster)
        return dict(label_counts)

    def merge_clusters(self, U, nc, class_hvs, batch_idx):
        pass

    def trim_clusters(self):
        pass