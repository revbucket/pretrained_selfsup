""" Helpful utilities for fine-tuning linear classifier on top of
    pretrained representation learning

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import pytorch_lightninng as pl


# ================================================================================
# =           Lightning Module to Contain Linear-Classifier components           =
# ================================================================================

class ResnetRepLinear(pl.LightningModule):
	def __init__(self, resnet, out_features=2, reset_linear=True, lr=3e-4):
		""" Takes in a pytorch resnet model (from torchvision.models.resnetXX)
		    and resets the linear layer (fc attribute of the resnet)
		"""
		super().__init__()
		self.resnet = resnet
		if reset_linear:
			self.resnet.fc = nn.Linear(resnet.fc.in_features, out_features)
		self.lr = lr

	def forward(self, x):
		return self.resnet(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		return {'loss': F.cross_entropy(y_hat, y)}

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		return {'loss': F.cross_entropy(y_hat, y)}


	def configure_optimizers(self):
		# Optimizer is the params from the SimCLR repo
		optimizer = optim.Adam(self.resnet.fc.parameters(), lr=self.lr, weight_decay=0.0008)



# ======================================================================================
# =           Handy method to do the full linear classifier training                   =
# ======================================================================================



def train_linear(resnet, train_loader, test_loader, num_classes=2, num_epochs=100, **kwargs):
	module = ResnetRepLinear(resnet, num_classes, reset_linear=True, **kwargs)
	trainer = pl.Trainer()

	# TODO: Add boilerplate pytorch training methods here