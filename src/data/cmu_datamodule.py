import pathlib

import numpy as np
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from numpy import random
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from mmsdk import mmdatasdk
from src.data.components.cmu_mosei_raw_dataset import CMUMoseiRawDataset
from transformers import BertTokenizer

from src.utils.data_helper import get_dominant_labels, cut_data


class CMUDataModule(LightningDataModule):
    train_dataset: CMUMoseiRawDataset = None
    val_dataset: CMUMoseiRawDataset = None
    test_dataset: CMUMoseiRawDataset = None

    default_dataloader_setting = dict(
        number_of_workers=0,
        pin_memory=False
    )

    def __init__(self,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._init_data()

    def create_vector_with_only_dominant_feature(self, emotion_label):
        features = emotion_label["features"][:]
        max_value = np.max(features)  # Get the maximum value
        dominant_indices = np.where(features == max_value)[0]  # Get indices of the dominant values

        # Handle multiple dominant values (if tie occurs)
        if len(dominant_indices) > 1:
            selected_index = random.choice(dominant_indices)  # Randomly pick one of the indices
        else:
            selected_index = dominant_indices[0]  # Only one dominant value

        # Update the array to keep only the dominant value
        max_vector = np.zeros_like(features)  # Reset the array to zeros or default values
        max_vector[0][selected_index] = max_value
        return max_vector

    def _init_data(self):
        project_folder = pathlib.Path(__file__).parent.parent.parent
        data_folder = project_folder / "data" / "final_aligned_raw"
        raw_dataset = mmdatasdk.mmdataset(str(data_folder))
        all_texts = []
        all_keys = []
        all_labels = []
        for (sentence_key, sentence), (label_key, label) in zip(
                raw_dataset.computational_sequences["words"].data.items(),
                raw_dataset.computational_sequences["All Labels"].data.items()):
            words = np.char.decode(sentence["features"]).astype(str)
            words = words[words != "sp"]
            all_texts.append(" ".join(list(words.ravel())))
            all_labels.append(self.create_vector_with_only_dominant_feature(label))
            all_keys.append((sentence_key, label_key))

        dominant_labels = get_dominant_labels(all_labels)
        train_labels, train_texts = cut_data(dominant_labels, all_texts)
        all_labels = np.array(train_labels)
        indices = np.arange(len(train_labels))
        all_texts = np.array(train_texts, dtype=object)
        np.random.shuffle(indices)

        train_texts = list(all_texts[indices][:round(0.7 * len(all_labels))])
        train_labels = all_labels[indices][:round(0.7 * len(all_labels))]
        val_texts = list(all_texts[indices][round(0.7 * len(all_labels)):round(0.8 * len(all_labels))])
        val_labels = all_labels[indices][round(0.7 * len(all_labels)):round(0.8 * len(all_labels))]
        test_texts = list(all_texts[indices][round(0.8 * len(all_labels)):])
        test_labels = all_labels[indices][round(0.8 * len(all_labels)):]

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_tokenized = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_tokenized = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
        test_tokenized = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

        self.train_texts = train_tokenized
        self.train_labels = train_labels
        self.val_texts = val_tokenized
        self.val_labels = val_labels
        self.test_texts = test_tokenized
        self.test_labels = test_labels

    def setup(self, stage: str) -> None:
        if stage == "validate":
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "train":
            if not self.train_dataset:
                self.train_dataset = CMUMoseiRawDataset(
                    tokenized_data=self.train_texts,
                    labels=self.train_labels
                )
            if not self.val_dataset:
                self.val_dataset = self._load_val()
        elif stage == "test":
            if not self.test_dataset:
                self.test_dataset = CMUMoseiRawDataset(
                    tokenized_data=self.test_texts,
                    labels=self.test_labels
                )

    def _load_val(self):
        return CMUMoseiRawDataset(
            tokenized_data=self.val_texts,
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
            shuffle=True,
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
    data_module = CMUDataModule(additional_var="xyz", batch_size=64, num_workers=0, pin_memory=False)
    data_module.setup("train")
    dataloader = data_module.train_dataloader()
    print(next(iter(dataloader)))
    print("DONE")
