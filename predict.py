#!/usr/bin/env python

"""
Loads the bootstrapped solution and runs the prediction test and optionally
generates a JSON report.
"""

from common import REPORT_DIR
from solution import args
from solution import data
from solution import data_set
from solution import ds_config
from solution import model
from solution import num_classes
from solution import test
from solution import train
import json
import math
import matplotlib.pyplot as plt
import os
import time


def observe(data_set, data, predictions):
  """Generates an observation report based on predictions."""
  num_classes = data_set.label_set.size
  error_matrix = [[0 for i in range(num_classes)] for i in range(num_classes)]
  report = {
    'results': [],
    'errors': [],
    'summary': {},
    'error_matrix': error_matrix,
    'data': data.info()
  }
  total = 0
  matched = 0
  noise_labels = set([
      'noise', 'circle', 'blank', '-circle', '-mixed-noise', '-pure-noise'
      ])
  for i, sample in enumerate(data.samples):
    image, label = sample
    actual_index = label.index
    predicted_index = int(predictions[i])
    predicted_label = data_set.label_set.labels()[predicted_index]
    actual_label = data_set.label_set.labels()[actual_index]
    predicted_is_noise = predicted_label.name in noise_labels
    actual_is_noise = actual_label.name in noise_labels
    observation = {
      'image': image.rel_path,
      'actual': label.name,
      'predicted': predicted_label.name,
      'matched': predicted_index == actual_index
    }
    total += 1
    if ((predicted_index == actual_index) or
        (predicted_is_noise and actual_is_noise)):
      matched += 1
    else:
      report['errors'].append(observation)
      error_matrix[predicted_index][actual_index] += 1
    report['results'].append(observation)

  accuracy = float(matched) / total
  report['summary'] = {
    'matched': matched,
    'total': total,
    'accuracy': accuracy,
    'display': "Matched %d of %d (%.2f%% accurate)" % (
          matched, total, accuracy * 100.0)
  }
  return report


def generate_report(data_set, data_prediction_pairs, config):
  train_config_filename = os.path.join(config.model_dir, 'config.json')
  train_config = {}
  if os.path.exists(train_config_filename):
    with open(train_config_filename) as f:
      train_config = json.load(f)
  report = {
    'id': time.strftime('%Y-%m-%d-%H%M%S'),
    'data_set': data_set.info(),
    'config': {
      'predict': config.__dict__,
      'train': train_config
    },
    'observations': []
  }
  for data, predictions in data_prediction_pairs:
    observations = observe(data_set, data, predictions)
    report['observations'].append(observations)
  return report


def save_report(report):
  if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR, 0o755)
  report_basename = report['id'] + '.json'
  report_filename = os.path.join(REPORT_DIR, report_basename)
  with open(report_filename, 'w') as f:
    json.dump(report, f, indent=2)


if __name__ == '__main__':
  if args.show_data_hash:
    print("Hash: all [ %s ] train [ %s ] test [ %s ]" % (
        data.hash(),
        train.hash(),
        test.hash()))

  if args.report:
    print("Generating report...")
    datas = [data, train, test]
    data_prediction_pairs = []
    for d in datas:
      sorted_data = d.sorted()
      predictions = list(model.predict(sorted_data.input_fn()))
      data_prediction_pairs.append((sorted_data, predictions))
    save_report(generate_report(data_set, data_prediction_pairs, args))

  data_eval = model.evaluate(data.input_fn())
  train_eval = model.evaluate(train.input_fn())
  test_eval = model.evaluate(test.input_fn())

  print("Accuracy: all [ %.2f ] train [ %.2f ] test [ %.2f ]" % (
      data_eval['accuracy'],
      train_eval['accuracy'],
      test_eval['accuracy']))
