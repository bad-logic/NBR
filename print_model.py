from keras.utils.vis_utils import plot_model
from keras.models import load_model

name = 'model1_aug.h5'
model = load_model(name)
plot_model(model, to_file='model.png',show_shapes=True)
