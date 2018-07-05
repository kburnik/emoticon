"""
Bootstraps and configures the input data and the model for use in actionable
code: training, evaluation and prediction.
"""

from dataset import DataSet
from dataset import DataSetConfig
from expander import expander
from model import build_model
import common

ds_config = DataSetConfig(
    num_channels=3,
    batch_size=20,
    expansion_factor=0,
    image_size=(64, 64),
    random_seed=271941)

data_set = DataSet(
    name="dataset",
    path=common.DATA_GENERATED_EXPANDED_20_DIR,
    config=ds_config)

num_classes = data_set.num_classes

data = data_set.data().shuffled()

train, test = data.split(0.6, names=["train", "test"])

model = build_model(
    ds_config,
    num_classes=num_classes,
    use_dropout=True,
    dropout_rate=0.5,
    learning_rate=0.01,
    momentum=0.6)
