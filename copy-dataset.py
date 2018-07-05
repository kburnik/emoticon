from solution import ds_config
from dataset import DataSet
import common

data_set = DataSet(
    name="dataset",
    path=common.DATA_COMMON_3_DIR,
    config=ds_config)

data_set_copy = data_set.copy(common.DATA_COMMON_3_DIR + "-grayscale")

for image, _ in data_set_copy.get_samples():
  image.transformed(lambda im: im.convert('L').convert('RGB')).save()


