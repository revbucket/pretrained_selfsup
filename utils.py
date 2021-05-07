""" Helpful utilities for fine-tuning linear classifier on top of
    pretrained representation learning

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# ================================================================================
# =           Lightning DataModule for Freezing Representations                  =
# ================================================================================
class FrozenRep(pl.LightningDataModule):
    """ Useful for when the dataset size is fairly small (can live in memory).
        Maps the given dataloader through the encoder and creates a new dataloader
    """
    def __init__(self, dataloader, encoder):
        """ Assumes encoder has last layer that's untrained/not useful """
        super().__init__()
        self.dataloader = dataloader
        self.headless_encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.x, self.y = [], []
        self.setup()

    def setup(self):
        for x, y in self.dataloader:
            with torch.no_grad():
                self.x.extend(self.headless_encoder(x))
                self.y.extend(y.numpy())
        self.x = torch.stack(self.x)
        self.y = torch.tensor(self.y)

        self.rep_data = torch.utils.data.TensorDataset(self.x, self.y)

    def train_dataloader(self, batch_size):
        return DataLoader(self.rep_data, batch_size)

    def test_dataloader(self, batch_size):
        return DataLoader(self.rep_data, batch_size)


# ================================================================================
# =           Lightning Module to Contain Linear-Classifier components           =
# ================================================================================

class LogisticReg(pl.LightningModule):
    """ Basic logistic regression. Use this when the dataset is
        (encoded_representations, labels)
    """
    def __init__(self, input_dim, output_dim, optim_kwargs=None):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        if optim_kwargs is None:
            optim_kwargs = {'lr': 3e-4}
        self.optim_kwargs = optim_kwargs


    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.max(dim=1)[1] == y).sum() / float(len(y))
        return {'test_loss': loss,
                'test_acc': acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.fc.parameters(), **self.optim_kwargs)
        return optimizer


class ResnetLinearHead(LogisticReg):
    """ Logistic Regression when the dataset is the images and needs to be
        mapped through the encoder first. As such, we just inherit and change
        the init/forward methods
    """
    def __init__(self, encoder, output_dim, optim_kwargs=None):
        # encoder has a last layer (untrained) named 'fc'
        input_dim = encoder.fc.in_features
        super().__init__(input_dim, output_dim, optim_kwargs=optim_kwargs)
        fc = nn.Linear(encoder.fc.in_features, output_dim)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1], fc)
        self.fc = fc

    def forward(self, x):
        with torch.no_grad():
            encx = self.encoder(x)
        return self.fc(encx.view(encx.shape[0], -1))


# ======================================================================================
# =           Handy method to do the full linear classifier training                   =
# ======================================================================================

def train_linear(encoder, trainloader, testloader, num_classes=2, num_epochs=500,
                 batch_size=256, freeze_reps=True):
    """
    encoder: the resnet encoder, the final fc layer is untrained
    trainloader/testloader: dataloaders for the _original_ dataset
    freeze_reps: if True, we make a dataset of frozen representations and train on that,
                 (much faster training)
    """
    if freeze_reps:
        trainloader = FrozenRep(trainloader, encoder).train_dataloader(batch_size)
        testloader = FrozenRep(testloader, encoder).test_dataloader(batch_size)
        x, y = next(iter(trainloader))
        input_dim = x.view(x.shape[0], -1).shape[1]
        model = LogisticReg(input_dim, num_classes)
    else:
        model = ResnetLinearHead(encoder, num_classes)

    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),
                         max_epochs=num_epochs)
    trainer.fit(model, trainloader)
    test_result = trainer.test(model, testloader)
    return {'model': model,
            'test_result': test_result}
