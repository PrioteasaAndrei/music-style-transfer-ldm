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


# Forward computation in LDM:
1. Content spectrogram (x) → Encoder → Latent (z_0)
2. Style spectrogram → StyleEncoder → Multi-resolution style embeddings
3. Forward Diffusion: (no trainable parameters)
   - Sample timestep t
   - Add noise to z_0: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
4. UNet prediction:
   - Input: z_t, t, style_embeddings
   - Output: predicted noise
5. Loss computation:
   - Diffusion loss: MSE(noise_pred, noise)
   - Autoencoder loss: MSE(x, reconstructed) + perceptual_loss + kl_loss
   - Style loss: (currently simple MSE, should be improved)


# TODOs Andrei

- [ ] style loss still doesnt change
- [ ] why isnt the decoder learning in the ldm

Debugging the style loss and decoder loss
- [x] check the data loader loads different spectograms
- [x] test dead style encoder



- [x] balance number of parameters of the two networks if one is much bigger the other will not learn
- [x] fix shape mismatch in the ldm
- [x] use pretrained LPIPS for perceptual loss during autoencoder training
- [x] implement DDIM during sampling
- [x] create different style encoder for style spectogram
- [x] decide on the number of parameters (256 x 256 images)
- [x] see if you integrade pytorch lightning
- [x] Implement skeleton for dataset
- [x] 1. Train autoencoder and decoder separately just for reconstruction (with perceptual and KL loss) - decoder should not be then frozen during the actual training just the encoder


# TODOS Theo

- [ ] listen to some songs from dataloader and reconstructed
- [ ] plot 4x4 spectograms from dataset and see how they look
- [ ] define a method / funciton that counts the number of params of the LDM class and the style encoder separately. (10M vs 3-4 M)

- [ ] Black Band problem.
- [ ] Why are the db in the reconstructed songs so different? Does changing normalization affect this? Can we just adjust volume?
- [ ] Statistics on the spectograms -> By label (maybe plot 16 spectograms per label) Do patterns emerge?
- [ ] Complete Dataset in different pytorch format


- [x] redo the dataset with saved images it is too slow now
