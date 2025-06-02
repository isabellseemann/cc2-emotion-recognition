import pathlib

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from mmsdk import mmdatasdk
from src.data.components.text_dataset import TextDataset
from src.utils.data_helper import get_dominant_labels, cut_data


class TextDataModule(LightningDataModule):
    train_dataset: TextDataset = None
    val_dataset: TextDataset = None
    test_dataset: TextDataset = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._init_data()

    def _init_data(self):
        project_folder = pathlib.Path(__file__).parent.parent.parent
        data_folder = project_folder / "data" / "aligned_glove_vectors_data"
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
            glove_vectors_dataset = mmdatasdk.mmdataset(str(data_folder))
            glove_tensors = glove_vectors_dataset.get_tensors(seq_len=50, non_sequences=["All Labels"], direction=False,
                                                          folds=[mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
                                                                 mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
                                                                 mmdatasdk.cmu_mosei.standard_folds.standard_test_fold])
            dominant_labels = get_dominant_labels(glove_tensors[0].get("All Labels"))
            self.train_labels, self.train_features = cut_data(dominant_labels, glove_tensors[0].get("glove_vectors"))
            self.train_features = torch.tensor(self.train_features, dtype=torch.float32)
            self.val_labels = glove_tensors[1].get("All Labels")
            self.val_features = glove_tensors[1].get("glove_vectors")
            self.test_labels = glove_tensors[2].get("All Labels")
            self.test_features = glove_tensors[2].get("glove_vectors")
            self.normalizer = Normalize(self.train_features.mean(0), self.train_features.std(0))
            self.val_features = self.normalizer(torch.tensor(self.val_features, dtype=torch.float32))
            self.test_features = self.normalizer(torch.tensor(self.test_features, dtype=torch.float32))
            self.train_features = self.normalizer(self.train_features)

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
                self.train_dataset = TextDataset(
                    glove_data=self.train_features,
                    labels=self.train_labels
                )
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "test":
            if not self.test_dataset:
                self.test_dataset = TextDataset(
                    glove_data=self.test_features,
                    labels=self.test_labels
                )

    def _load_val(self):
        return TextDataset(
            glove_data=self.val_features,
            labels=self.val_labels
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == '__main__':
    data_module = TextDataModule(batch_size=64, num_workers=0, pin_memory=False)
    data_module.setup("train")
    dataloader = data_module.train_dataloader()
    print(next(iter(dataloader)))
    print("DONE")
