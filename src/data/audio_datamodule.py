import pathlib

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from mmsdk import mmdatasdk
from src.data.components.video_and_audio_dataset import AudioAndVideoDataset
from src.utils.data_helper import get_dominant_labels, cut_data
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    batch: list of tuples (features, label)
    - features: tensor (seq_len, feature_dim)
    - label: scalar tensor
    Returns:
    - padded_features: (batch_size, max_seq_len, feature_dim)
    - labels: (batch_size,)
    - lengths: (batch_size,) original sequence lengths before padding
    """
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["labels"].item() for item in batch], dtype=torch.long)
    lengths = torch.tensor([f.size(0) for f in features])
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    return {
        "features": padded_features,
        "lengths": lengths,
        "labels": labels
    }


class AudioDataModule(LightningDataModule):
    train_dataset: AudioAndVideoDataset = None
    val_dataset: AudioAndVideoDataset = None
    test_dataset: AudioAndVideoDataset = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._init_data()

    def _init_data(self):
        project_folder = pathlib.Path(__file__).parent.parent.parent
        data_folder = project_folder / "data" / "aligned_covarep_data"
        train_data_file = data_folder / "train_data.pkl"
        train_labels_file = data_folder / "train_labels.pkl"
        val_data_file = data_folder / "val_data.pkl"
        val_labels_file = data_folder / "val_labels.pkl"
        test_data_file = data_folder / "test_data.pkl"
        test_labels_file = data_folder / "test_labels.pkl"

        if (train_data_file.exists() and train_labels_file.exists()
            and val_data_file.exists() and val_labels_file.exists()
            and test_data_file.exists() and test_labels_file.exists()
        ):
            with open(train_data_file, "rb") as f:
                self.train_features = torch.load(f)
            with open(train_labels_file, "rb") as f:
                self.train_labels = torch.load(f)
            with open(val_data_file, "rb") as f:
                self.val_features = torch.load(f)
            with open(val_labels_file, "rb") as f:
                self.val_labels = torch.load(f)
            with open(test_data_file, "rb") as f:
                self.test_features = torch.load(f)
            with open(test_labels_file, "rb") as f:
                self.test_labels = torch.load(f)
            return
        else:
            covarep_dataset = mmdatasdk.mmdataset(str(data_folder))
            audio_tensors = covarep_dataset.get_tensors(seq_len=50, non_sequences=["All Labels"], direction=False,
                                                        folds=[mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
                                                               mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
                                                               mmdatasdk.cmu_mosei.standard_folds.standard_test_fold])
            dominant_labels = get_dominant_labels(audio_tensors[0].get("All Labels"))
            self.train_labels, self.train_features = cut_data(dominant_labels, audio_tensors[0].get("COVAREP"))
            self.train_features = torch.tensor(self.train_features, dtype=torch.float32)
            self.val_labels = audio_tensors[1].get("All Labels")
            self.val_features = torch.tensor(np.array(audio_tensors[1].get("COVAREP")), dtype=torch.float32)
            self.test_labels = audio_tensors[2].get("All Labels")
            self.test_features = torch.tensor(np.array(audio_tensors[2].get("COVAREP")), dtype=torch.float32)

            with open(train_data_file, "wb") as f:
                torch.save(self.train_features, f)
            with open(train_labels_file, "wb") as f:
                torch.save(torch.tensor(self.train_labels, dtype=torch.float32), f)
            with open(val_data_file, "wb") as f:
                torch.save(self.val_features, f)
            with open(val_labels_file, "wb") as f:
                torch.save(torch.tensor(self.val_labels, dtype=torch.float32), f)
            with open(test_data_file, "wb") as f:
                torch.save(self.test_features, f)
            with open(test_labels_file, "wb") as f:
                torch.save(torch.tensor(self.test_labels, dtype=torch.float32), f)

    def setup(self, stage: str) -> None:
        if stage == "validate":
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "train":
            if not self.train_dataset:
                self.train_dataset = AudioAndVideoDataset(
                    features_list=self.train_features,
                    labels_list=self.train_labels,
                )
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "test":
            if not self.test_dataset:
                self.test_dataset = AudioAndVideoDataset(
                    features_list=self.test_features,
                    labels_list=self.test_labels,
                )

    def _load_val(self):
        return AudioAndVideoDataset(
            features_list=self.val_features,
            labels_list=self.val_labels,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )


if __name__ == '__main__':
    data_module = AudioDataModule(batch_size=64, num_workers=7, pin_memory=False)
    data_module.setup("train")
    dataloader = data_module.train_dataloader()
    print(next(iter(dataloader)))
    print("DONE")
