from typing import Any

import numpy as np
import seaborn as sns
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchmetrics import Accuracy, MaxMetric, F1Score, ConfusionMatrix

from src.models.components.classifier import Classifier
from src.models.components.emotion_embedding_creator import EmotionClassifier
from src.models.components.video_emotion_embedding_creator import AudioVideoEmotionEmbeddingCreator


class CmuMoseiModule(LightningModule):
    def __init__(self, model: EmotionClassifier,
                 optimizer: torch.optim.Optimizer,
                 learning_rate,
                 compile: bool,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=7)
        self.val_acc = Accuracy(task="multiclass", num_classes=7)
        self.test_acc = Accuracy(task="multiclass", num_classes=7)
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.val_confmat_best = MaxMetric()
        self.val_f1 = F1Score(num_classes=7, task="multiclass")
        self.val_confmat = ConfusionMatrix(num_classes=7, task="multiclass")
        self.test_f1 = F1Score(num_classes=7, task="multiclass")
        self.test_confmat = ConfusionMatrix(num_classes=7, task="multiclass")

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def on_train_start(self) -> None:
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_f1_best.reset()
        self.val_confmat.reset()
        self.val_f1.reset()

    def on_train_epoch_start(self) -> None:
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_f1_best.reset()
        self.val_confmat.reset()
        self.val_f1.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def model_step(self, batch: dict):
        labels = batch["labels"]
        predictions = self(**{key: value for key, value in batch.items() if key != "labels"})
        loss = self.criterion(predictions, labels)

        return loss, predictions.argmax(1), labels

    def training_step(self, batch, batch_index) -> Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_index) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_acc(preds, targets)
        self.val_confmat(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=True)
        # self.log("val/confmat", self.val_confmat, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        confmat = self.val_confmat.compute()
        self.val_acc_best(acc)
        self.val_f1_best(f1)
        # self.val_confmat_best(confmat)
        # best_val_confmat = self.val_confmat_best.compute()

        # logger.log_metrics({"val/confusion": self.val_confmat.confmat})
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                fig, ax = plt.subplots(figsize=(15, 10))
                confmat = confmat.cpu()
                annot = np.array([[f"{int(x):,}" for x in row] for row in confmat])
                sns.heatmap(confmat, annot=annot, ax=ax,
                            fmt='', cmap=sns.light_palette("#79C", as_cmap=True))
                logger.log_image("val/confusion-matrix", [wandb.Image(fig)])
        # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # conf_matrix = self.val_confmat.compute()
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/best_F1", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/best_confmat", best_val_confmat, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_acc(preds, targets)
        self.test_confmat(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_epoch=True, prog_bar=True)


if __name__ == '__main__':
    model = AudioVideoEmotionEmbeddingCreator(7, cnn_out_channels=64, lstm_hidden_size=128, lstm_layers=1, dropout=0.3,
                                              classifier_hidden_size=128)
    classifier = EmotionClassifier(model, Classifier(64))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0)
    model = CmuMoseiModule(
        classifier,
        optimizer=optimizer,
        learning_rate=0.002,
        compile=False
    )
    print("DONE")
