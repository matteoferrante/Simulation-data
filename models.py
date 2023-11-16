import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from torchmetrics import Accuracy


class BaseModule(pl.LightningModule):
    def __init__(self, model, num_classes, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # initialize the model
        self.model = model

        # store the optimization hyperparameter
        self.lr = lr

        # initialize helper metric objects
        self.train_acc = Accuracy(num_classes=num_classes)
        self.val_acc = Accuracy(num_classes=num_classes)
        self.test_acc = Accuracy(num_classes=num_classes)
    
    def forward(self, x):        
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = self(x)

        loss = F.cross_entropy(x, y)

        x = F.softmax(x, dim=-1)
        
        self.train_acc(x, y)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x = self(x)
        x = F.softmax(x, dim=-1)
        
        self.val_acc(x, y)

        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        x = self(x)
        x = F.softmax(x, dim=-1)
        
        self.test_acc(x, y)

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)