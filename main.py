import load_data
from model import NVIDA

nvidia = NVIDA()
train_x, train_y, test_x, test_y = load_data.return_data()

nvidia.fit(train_x, train_y,
    validation_data=(test_x, test_y),
    nb_epoch=1
)
