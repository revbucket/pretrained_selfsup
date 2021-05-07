# Pretrained Self Supervised Models (pytorch)

I've Collected several pretrained pytorch models for easy prototyping on different datasets.
None of these models are my own, but here's the githubs from which I've collected them. 
- [SimCLR](https://github.com/AndrewAtanov/simclr-pytorch)
- [MocoV2](https://github.com/facebookresearch/moco)
- [SwAV](https://github.com/facebookresearch/swav)
- [BYOL](https://github.com/sthalles/PyTorch-BYOL)
These model weights can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1FLcG_6prgQ2oA_aqKpzTEj_lTFx1n8p7/view?usp=sharing)

These models are included in `model_zoo.py`, along with some extra details about architecture and training method. 
A handy method for training a linear probe on top of a pretrained representation is contained in `utils.py'
