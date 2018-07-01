from model import DropoutMode
from model import ActivationMode
from model import build_model

from dataset import DataSet
from dataset import DataSetConfig

import common

def expander(sample):
  import random
  from PIL import Image as PilImage
  image, label = sample

  def rand_offset(size):
    return round(random.random() * size * 2 - size)

  def translate(image):
    return image.convert('RGB').transform(
        image.size,
        PilImage.AFFINE,
        (
          1,
          0,
          rand_offset(5),
          0,
          1,
          rand_offset(5)
        ),
        fill=1,
        fillcolor="white")

  def rotate(image):
    rot = image.rotate(rand_offset(25), expand=1)
    fff = PilImage.new('RGBA', rot.size, (255,)*4)
    return PilImage.composite(rot, fff, rot)

  for i in range(20):
    new_image = image.transformed(translate)
    # new_image.image.show()
    # input("next? ")
    yield (new_image, label)

ds_config = DataSetConfig(
    max_labels=6,
    num_channels=3,
    batch_size=8,
    image_size=(64, 64),
    random_seed=3171945)

ds = DataSet(
    name="simple",
    path=common.DATA_SIMPLE_DIR,
    config=ds_config)

data = ds.data().shuffled()

train, test = data.split(0.7, names=["train", "test"])

train = train.expanded(expander).shuffled()
test = test.expanded(expander).shuffled()

print("train num labels", len(train.labels()))
print("train data shape", train.images().shape)

print("test num labels", len(test.labels()))
print("test data shape", test.images().shape)


model = build_model(
    ds_config,
    num_classes=ds.label_set.size,
    learning_rate=0.01,
    momentum=0.7)


while True:
  model.train(train.input_fn(), steps=1000)
  evaluation = model.evaluate(test.input_fn())
  print("Test accuracy", evaluation['accuracy'])
