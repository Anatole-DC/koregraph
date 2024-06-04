from tensorflow.python.keras.callbacks import ModelCheckpoint

from koregraph.params import WEIGHTS_BACKUP_DIRECTORY

BackupCallback = ModelCheckpoint(
    WEIGHTS_BACKUP_DIRECTORY,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)
