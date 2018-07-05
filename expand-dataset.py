from solution import ds_config
from dataset import DataSet
from expander import expander
import common
from PIL import Image as PilImage

data_set = DataSet(
    name="dataset",
    path=common.DATA_GENERATED_DIR,
    config=ds_config)

expansion_factor = 20

data_set_copy = data_set.copy(
    common.DATA_GENERATED_DIR + ("-expanded-%d" % expansion_factor))

for original_image, _ in data_set_copy.get_samples():
  for new_image, _ in expander((original_image, _), expansion_factor):
    print("Saving", new_image.path)
    im = new_image.read(image_size=ds_config.image_size, num_channels=3)
    im.save(new_image.path, 'png')
  original_image.remove()
