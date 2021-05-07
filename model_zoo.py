""" Techniques to load common pretrained encoders.
	Will cite/credit as much as possible in each section
"""

import torch
import torch.nn as nn
import torchvision
import glob
from collections import OrderedDict
# =======================================================================
# =                          SimCLR                                     =
# =======================================================================
"""
SimCLR:
Github: https://github.com/google-research/simclr
	Arch: Resnet50
	Dataset: Imagenet (top1 accuracy of 71.7)
	Num Epochs: 600
	Batch Size: 2k
"""


SIMCLR_PATH = 'models/simclr_resnet50_bs2k_epochs600.pth.tar'
def load_pretrained_simclr():
    state_dict = torch.load(SIMCLR_PATH, map_location='cpu')['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        PREFIX = 'convnet.'
        if k.startswith(PREFIX):
            new_k = k[len(PREFIX):]
            new_state_dict[new_k] = v
    resnet50 = torchvision.models.resnet50()
    resnet50.load_state_dict(new_state_dict)
    return resnet50



# =======================================================================
# =                          MoCoV2                                     =
# =======================================================================
"""
MoCoV2
Github: https://github.com/facebookresearch/moco
	Arch: Resnet50
	Dataset: Imagenet (top1 accuracy of 71.1)
	Num EpochS: 800
"""


MOCO_PATH = 'models/mocov2_epochs800.pth.tar'
def load_pretrained_moco():
    resnet = torchvision.models.resnet50()
    checkpoint = torch.load(MOCO_PATH, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = resnet.load_state_dict(state_dict, strict=False)
    return resnet




# =======================================================================
# =                          SwAV                                       =
# =======================================================================
"""
SwAV
Github: https://github.com/facebookresearch/swav
	Arch: Resnet50
	Dataset: Imagenet (top1 accuracy of 75.3)
	Num epochs: 800
	BatchSize: 4k
"""


def load_pretrained_swav():
    swav = torch.hub.load('facebookresearch/swav', 'resnet50')
    return swav.train()

# =======================================================================
# =                          BYOL                                       =
# =======================================================================
"""
BYOL
Github: https://github.com/sthalles/PyTorch-BYOL [not official implementation]
	Arch: Resnet18
	Dataset: STL10 (top1 accuracy of 75.2)
	Num Epochs: 80
	Batch Size: 64
"""

BYOL_PATH = 'models/byol_stl10_bs64_epochs80.pth.tar'

def load_pretrained_byol():

	loader_args = {'name': 'resnet18',
	               'projection_head':
	               		{'mlp_hidden_size': 512,
	               		 'projection_size': 128}}
	state_dict = torch.load(BYOL_PATH, map_location='cpu')['online_network_state_dict']
	resnet18 = torchvision.models.resnet18(pretrained=False)

	# Hacky nonsense to not use sthalles' custom classes

	new_state_dict = OrderedDict()
	for ((k_old, v_old), (k_van, v_van)) in zip(state_dict.items(), resnet18.state_dict().items()):
		if (v_old.shape == v_van.shape) and k_old.startswith('encoder'):
			new_state_dict[k_van] = v_old

	resnet18.load_state_dict(new_state_dict, strict=False)
	return resnet18.train()
