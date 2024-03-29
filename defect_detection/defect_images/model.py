from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassRecall


import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import nn
import torchvision.models as models

from torchvision.ops import sigmoid_focal_loss

class ARMBenchImageDefectModule(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """ARMBenchImageDefectModule.
        Args:
            lr: Learning rate
            lr_scheduler_gamma: Factor by which the learning rate
        """
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.num_classes = 3

        # Define model and load pretrained weights
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_classes)

        # Loss function. 
        self.loss_func = sigmoid_focal_loss

        # Define metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes )
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes )
        self.train_recall = MulticlassRecall(num_classes=self.num_classes )
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes )
        self.acc = Accuracy(task='multiclass', num_classes=self.num_classes )
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes = self.num_classes).float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y_one_hot, reduction='mean')
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc(y_hat, y_one_hot))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes = self.num_classes).float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y_one_hot, reduction='mean')
        self.log("val_acc", self.val_acc(y_hat, y_one_hot))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes = self.num_classes).float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y_one_hot, reduction='mean')
        self.log("test_loss", loss)
        self.log("test_acc", self.acc(y_hat, y_one_hot))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
