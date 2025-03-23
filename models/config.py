
config = {
    'learning_rate': 5e-3,
    'learning_rate_factor': 0.5,
    'learning_rate_patience': 5,
    'learning_rate_min': 1e-6,
    'num_epochs': 50,
    # 'batch_size': 256,
    'batch_size': 128,
    'style_loss_weight': 3.0,
    'latent_dim_encoder': 32,
    'data_dir': 'downloads/',
    'processed_spectograms_dataset_folderpath': 'processed_images',
    'pairing_file_path': 'spectrogram_pair_dataset_pairings.csv',
    'unet_num_filters': 64,
    'forward_diffusion_num_timesteps': 1000,
    'compression_feature_extractor': 'vggish',
}
