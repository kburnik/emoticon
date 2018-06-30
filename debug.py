
from dataset import DataSet
from dataset import DataSetConfig
import common
import os

if __name__ == '__main__':
  ds_config = DataSetConfig(
      max_labels=6,
      num_channels=3,
      batch_size=8,
      image_size=(32, 32),
      random_seed=None)

  dataset = DataSet(
      name="dummy",
      path=common.DATA_DUMMY_DIR,
      config=ds_config)

  dataset.print_info()

  # dataset_copy = dataset.copy(os.path.join(common.DATA_ROOT_DIR, 'work'), force=True)
  # dataset_copy.print_info()

  data = dataset.data().shuffled()

  train, test = data.split(0.7, names=["train", "test"])

  images = data.images()
  labels = data.labels()

  print(data.name, data.labels())
  print(train.name, train.labels())
  print(test.name, test.labels())


