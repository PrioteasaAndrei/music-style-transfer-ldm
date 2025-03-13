from model import *
from loss import *


def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    noise_scheduler = ForwardDiffusion(num_timesteps=100)














def main():
    train()

if __name__ == "__main__":
    main()