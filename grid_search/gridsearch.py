
param_grid = {
    'seq_len': [1, 3, 6, 9, 12],
    'n_block': [1, 2, 3],
    'dropout': [0.1, 0.3, 0.5, 0.7, 0.9],
    'batch_size': [1, 2, 3, 4],
    'ff_dim': [8, 16, 32, 64, 128],
    'learning_rate': [1e-4, 1e-5]
}