
config = {
    'learning_rate': 1e-4,
    'learning_rate_factor': 0.5,
    'learning_rate_patience': 5,
    'learning_rate_min': 1e-6,
    'num_epochs': 50,
    'batch_size': 32,
    'style_loss_weight': 0.1,
    'latent_dim_encoder': 32,
    'data_dir': 'downloads/',
    'file_name': 'processed_dataset.parquet',
    'unet_num_filters': 64,
}
