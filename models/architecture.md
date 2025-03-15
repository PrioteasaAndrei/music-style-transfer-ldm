## LDM components
1. VAE encoder
2. VAE decoder 
3. Denoising model: UNet / Resnet with cross attention
4. Noise scheduler: DDIM


## Paper

Original LDM paper has 200-300M params

Encoder: pretrained

Centre latent space with tanh in [-1,1]

UNet:  14 layer convolution blocks and atetntion blocks (in their paper)

Forward noising: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$