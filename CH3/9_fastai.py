from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)

data = ImageDataBunch.from_folder(path)

learner = create_cnn(data, models.resnet18, metrics=accuracy)
learner.fit(1)
