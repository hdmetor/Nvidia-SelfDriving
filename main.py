import load_data
from model import NVIDA

from keras.callbacks import ModelCheckpoint

print('Loading data...')
train_x, train_y, test_x, test_y = load_data.return_data()


print('Loading model')
nvidia = NVIDA()

checkpointer = ModelCheckpoint(
    filepath="weights_dropout.hdf5",
    verbose=1,
    save_best_only=True
)
epochs = 100
batch_size = 128

print('Starting training')
nvidia.fit(train_x, train_y,
    validation_data=(test_x, test_y),
    nb_epoch=epochs,
    batch_size=batch_size,
    callbacks=[checkpointer]
)

print('Done')
