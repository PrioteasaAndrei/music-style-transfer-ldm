# Description

This is the project based on <https://arxiv.org/pdf/2404.14771>, for the final project in the GNN class of WS24/25, Faculty of Informatik, University of Heidelberg.

Members: Prioteasa, Cristi Andrei & Stempel-Hauburger, Theo

This project implements the framework described in the paper Music Style Transfer With Diffusion Model by Hong Huang, Yuyi Wang, Luyao Li, and Jun Lin. The paper addresses limitations in previous style transfer methods by introducing a spectrogram-based approach using diffusion models for multi-style music transfer. The proposed method leverages the GuideDiff technique to improve audio restoration, reducing noise and enabling high-quality real-time audio generation on consumer-grade GPUs.

# Tricks

- try to overfit on a single sample to debug your pipeline (train and eval on a single sample)
- use learning rate scheduler (e.g. cosine)
- definetly used normalization layers (if not explicitly present in architecture)
- maybe integrate <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>
- OmegaConf for configs: <https://github.com/ashleve/lightning-hydra-template>

# TODOs Andrei
AUTOENCODER IS NOT REALLY WORKING
CHECK IF IT THE LATENT SPACE - train with new latent space (32) and check the test for improvements
- [ ] remove lpips and see what changes
- [ ] fix shape mismatch in the ldm
- [ ] check that forward diffusion and ddim are correct
- [ ] see if KL regularization is actually needed 

- [x] use pretrained LPIPS for perceptual loss during autoencoder training
- [x] implement DDIM during sampling
- [x] create different style encoder for style spectogram
- [x] decide on the number of parameters (256 x 256 images)
- [x] see if you integrade pytorch lightning
- [x] Implement skeleton for dataset
- [x] 1. Train autoencoder and decoder separately just for reconstruction (with perceptual and KL loss) - decoder should not be then frozen during the actual training just the encoder


# TODOS Theo

- [ ] Black Band problem.
- [ ] How to get a more fixed size image? -> 256x256?
- [ ] image normalization?ß
- [ ] Why are the db in the reconstructed songs so different? Does changing normalization affect this? Can we just adjust volume?
- [ ] Statistics on the spectograms -> By label (maybe plot 16 spectograms per label) Do patterns emerge?
- [ ] Complete Dataset in different pytorch format


- [x] redo the dataset with saved images it is too slow now
