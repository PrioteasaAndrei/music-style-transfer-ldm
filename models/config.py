
config = {
    'learning_rate': 1e-4,
    'learning_rate_factor': 0.5,
    'learning_rate_patience': 5,
    'learning_rate_min': 1e-6,
    'num_epochs': 1000,
    'batch_size': 32,
    'style_loss_weight': 0.1,
    'latent_dim_encoder': 32,
    'data_dir': 'downloads/',
    'processed_spectograms_dataset_folderpath': 'processed_images',
    'pairing_file_path': 'spectrogram_pair_dataset_pairings.csv',
    'unet_num_filters': 64,
}
