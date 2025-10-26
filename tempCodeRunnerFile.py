        checkpoint = torch.load(MODEL_DIR, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.feat_extract.load_state_dict(checkpoint['state_dict'])
        else:
            self.feat_extract.load_state_dict(checkpoint)
        self.feat_extract.eval()