import load_data
from model import NVIDA

from keras.callbacks import ModelCheckpoint

nvidia = NVIDA()
train_x, train_y, test_x, test_y = load_data.return_data()


checkpointer = ModelCheckpoint(
    filepath="weights.hdf5",
    verbose=1,
    save_best_only=True
)
epochs = 100
batch_size = 128

nvidia.fit(train_x, train_y,
    validation_data=(test_x, test_y),
    nb_epoch=epochs,
    batch_size=batch_size,
    callbacks=[checkpointer]
)
