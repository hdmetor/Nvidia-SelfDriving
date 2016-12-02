import load_data
import model

train_x, train_y, test_x, test_y = load_data.return_data()

model = model.NVIDA()

model.fit(train_x, train_y,
    validation_data=(test_x, test_y),
    nb_epoch=10
)
