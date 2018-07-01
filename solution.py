from dataset import DataSet
from dataset import DataSetConfig
from expander import expander
from model import build_model
import common

ds_config = DataSetConfig(
    max_labels=6,
    num_channels=3,
    batch_size=8,
    image_size=(64, 64),
    expansion_factor=10,
    random_seed=3171945)

data_set = DataSet(
    name="simple",
    path=common.DATA_SIMPLE_DIR,
    config=ds_config)

num_classes = data_set.num_classes

data = data_set.data().shuffled()

model = build_model(
    ds_config,
    num_classes=num_classes,
    learning_rate=0.01,
    momentum=0.7)
