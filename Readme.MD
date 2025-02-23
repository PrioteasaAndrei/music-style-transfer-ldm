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

# TODOS

- [ ] way to add todos
