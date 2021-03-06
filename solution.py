"""
Bootstraps and configures the input data and the model for use in actionable
code: training, evaluation and prediction.
"""

from common import DataPath
from config import parse_config
from dataset import DataSet
from dataset import DataSetConfig
from expander import expander
from model import build_model


args = parse_config()

ds_config = DataSetConfig(
    num_channels=args.num_channels,
    monochrome=args.monochrome,
    batch_size=args.batch_size,
    expansion_factor=args.expansion_factor,
    image_size=(args.image_size, args.image_size),
    random_seed=args.random_seed)

data_set = DataSet(
    name=args.dataset,
    path=DataPath.get(args.dataset),
    config=ds_config)

num_classes = data_set.num_classes

data = data_set.data().shuffled()

train, test = data.split(args.split_ratio, names=["train", "test"])

model = build_model(
    # Dataset.
    ds_config=ds_config,

    # Input.
    num_classes=num_classes,

    # Options from flags.
    args=args)
