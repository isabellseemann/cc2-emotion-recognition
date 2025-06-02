import pathlib

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.data.components.trimodal_dataset import TrimodalDataset


class TrimodalDataModule(LightningDataModule):
    train_dataset: TrimodalDataset = None
    val_dataset: TrimodalDataset = None
    test_dataset: TrimodalDataset = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._init_data()

    def _init_data(self, *args, **kwargs):
        project_folder = pathlib.Path(__file__).parent.parent.parent
        data_folder = project_folder / "data" / "aligned_open_face_data"
        train_data_file = data_folder / "train_data.pkl"
        train_labels_file = data_folder / "train_labels.pkl"
        val_data_file = data_folder / "val_data.pkl"
        val_labels_file = data_folder / "val_labels.pkl"
        test_data_file = data_folder / "test_data.pkl"
        test_labels_file = data_folder / "test_labels.pkl"

        with open(train_data_file, "rb") as f:
            self.train_video_features = torch.load(f)
        with open(train_labels_file, "rb") as f:
            self.train_labels = torch.load(f)
        with open(val_data_file, "rb") as f:
            self.val_video_features = torch.load(f)
        with open(val_labels_file, "rb") as f:
            self.val_labels = torch.load(f)
        with open(test_data_file, "rb") as f:
            self.test_video_features = torch.load(f)
        with open(test_labels_file, "rb") as f:
            self.test_labels = torch.load(f)

        data_folder = project_folder / "data" / "aligned_covarep_data"
        train_data_file = data_folder / "train_data.pkl"
        val_data_file = data_folder / "val_data.pkl"
        test_data_file = data_folder / "test_data.pkl"

        with open(train_data_file, "rb") as f:
            self.train_audio_features = torch.load(f)
        with open(val_data_file, "rb") as f:
            self.val_audio_features = torch.load(f)
        with open(test_data_file, "rb") as f:
            self.test_audio_features = torch.load(f)

        data_folder = project_folder / "data" / "aligned_glove_vectors_data"
        train_data_file = data_folder / "train_data.pkl"
        val_data_file = data_folder / "val_data.pkl"
        test_data_file = data_folder / "test_data.pkl"

        with open(train_data_file, "rb") as f:
            self.train_text_features = torch.load(f)
        with open(val_data_file, "rb") as f:
            self.val_text_features = torch.load(f)
        with open(test_data_file, "rb") as f:
            self.test_text_features = torch.load(f)

    def setup(self, stage: str) -> None:
        if stage == "validate":
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "train":
            if not self.train_dataset:
                self.train_dataset = TrimodalDataset(
                    audio_features=torch.tensor(self.train_audio_features, dtype=torch.float32),
                    video_features=torch.tensor(self.train_video_features, dtype=torch.float32),
                    text_features=torch.tensor(self.train_text_features, dtype=torch.float32),
                    labels=self.train_labels
                )
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "test":
            if not self.test_dataset:
                self.test_dataset = TrimodalDataset(
                    audio_features=torch.tensor(self.test_audio_features, dtype=torch.float32),
                    video_features=torch.tensor(self.test_video_features, dtype=torch.float32),
                    text_features=torch.tensor(self.test_text_features, dtype=torch.float32),
                    labels=self.test_labels
                )

    def _load_val(self):
        return TrimodalDataset(
            audio_features=torch.tensor(self.val_audio_features, dtype=torch.float32),
            video_features=torch.tensor(self.val_video_features, dtype=torch.float32),
            text_features=torch.tensor(self.val_text_features, dtype=torch.float32),
            labels=self.val_labels
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

if __name__ == '__main__':
    data_module = TrimodalDataModule(batch_size=64, num_workers=0, pin_memory=False)
    data_module.setup("train")
    dataloader = data_module.train_dataloader()
    print(next(iter(dataloader)))
    print("DONE")
