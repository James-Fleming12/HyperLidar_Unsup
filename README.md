# Test Repo for Self Supervised Online HyperLiDAR

current test is in exp.py

dataset needs to be contained in /kitti_data/sequences

possible configurations for high-level layer:

1. Only Pooling (perfect averaging but not meaningful for confidence)
2. Pooling then Linear (tradeoff of computation and interpretability?)

Model saved in extractor_model.pth with loss of 0.7588