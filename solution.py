from dataset import DataSet
from dataset import DataSetConfig
from expander import expander
from model import build_model
import common

ds_config = DataSetConfig(
    max_labels=6,
    num_channels=3,
    batch_size=12,
    image_size=(64, 64),
    expansion_factor=20,
    random_seed=3171945)

data_set = DataSet(
    name="common-6",
    path=common.DATA_COMMON_6_DIR,
    config=ds_config)

num_classes = data_set.num_classes

data = data_set.data().shuffled()

train, test = data.split(0.6, names=["train", "test"])

train_expanded = train.expanded(expander).shuffled()

model = build_model(
    ds_config,
    num_classes=num_classes,
    use_dropout=False,
    learning_rate=0.005,
    momentum=0.7)
