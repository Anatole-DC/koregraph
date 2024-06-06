from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from koregraph.params import WEIGHTS_BACKUP_DIRECTORY

BackupCallback = ModelCheckpoint(
    WEIGHTS_BACKUP_DIRECTORY / 'backup.keras',
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)

StoppingCallback = EarlyStopping(
    monitor="val_loss",
    patience=2,
    verbose=0,
    restore_best_weights=True
)
