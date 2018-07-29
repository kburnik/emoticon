#!/usr/bin/env python

"""
Loads the bootstrapped solution and runs the training.
"""

from expander import expander
from solution import args
from solution import data
from solution import data_set
from solution import model
from solution import test
from solution import train
from visual import display_data_detached
import json
import os

train_expanded = train.expanded(expander).shuffled()

print("Data shape", data.images().shape)
print("Num classes", data_set.num_classes)
print("Data size: train [ %d ] train_expanded [ %d ] test [ %d ]" % (
    train.size,
    train_expanded.size,
    test.size))

if args.show_data:
  display_data_detached(train_expanded, test)

# Store the used training configuration.
if not os.path.exists(args.model_dir):
  os.makedirs(args.model_dir, 0o755)
with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
  json.dump(args.__dict__, f, indent=2)

i = 0
last_accuracy_stat = [0, 0, 0]
accuracy_stat_repeated = 0
while True:
  i += 1
  model.train(train_expanded.input_fn(), steps=args.training_steps)

  train_eval = model.evaluate(train.input_fn())
  train_expanded_eval = model.evaluate(train_expanded.input_fn())
  test_eval = model.evaluate(test.input_fn())

  print(
      "%03d Accuracy: train [ %.2f ] train_expanded [ %.2f ] test [ %.2f ]" % (
          i,
          train_eval['accuracy'],
          train_expanded_eval['accuracy'],
          test_eval['accuracy']))

  evals = [train_eval, train_expanded_eval, test_eval]
  accuracy_stat = [e['accuracy'] for e in evals]
  if accuracy_stat == last_accuracy_stat
    accuracy_stat_repeated += 1
  else:
    accuracy_stat_repeated = 0
  last_accuracy_stat = accuracy_stat

  # Stop when converged to validation set and did at least 5 iterations.
  # Also stop when the stats repeat 3 times or did over 50 iterations.
  if ((train_eval['accuracy'] > 0.999 and i > 5) or
      accuracy_stat_repeated > 2 or
      i > 50):
    break

