import torch
from torch.utils.data import Dataset


class AudioAndVideoDataset(Dataset):
    def __init__(self, features_list, labels_list):
        """
        features_list: list of numpy arrays, each shape (num_frames, feature_dim)
        labels_list: list or numpy array of labels (int) per sequence
        """
        self.features_list = features_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        feat = self.features_list[idx]

        label = self.labels_list[idx]

        if not torch.is_tensor(feat):
            feat = torch.tensor(feat, dtype=torch.float32)

        # Convert one-hot to integer label
        label = torch.tensor(label, dtype=torch.float32)
        class_index = int(label.argmax())  # scalar int
        class_index = torch.tensor(class_index, dtype=torch.long)

        return {
            'features': feat,
            'labels': class_index,
            'lengths': torch.tensor(feat.size(0))
        }