"""
Loads the bootstrapped solution and runs the training.
"""

from expander import expander
from solution import data
from solution import data_set
from solution import model
from solution import test
from solution import train

print("Expanding data set.")
train_expanded = train.shuffled() # .expanded(expander)

train_expanded.sorted().show()
test.sorted().show()

print("Data shape", data.images().shape)
print("Num classes", data_set.num_classes)
print("Data size: train [ %d ] train_expanded [ %d ] test [ %d ]" % (
    train.size,
    train_expanded.size,
    test.size))

while True:
  model.train(train_expanded.input_fn(), steps=100)

  train_eval = model.evaluate(train.input_fn())
  train_expanded_eval = model.evaluate(train_expanded.input_fn())
  test_eval = model.evaluate(test.input_fn())

  print("Accuracy: train [ %.2f ] train_expanded [ %.2f ] test [ %.2f ]" % (
      train_eval['accuracy'],
      train_expanded_eval['accuracy'],
      test_eval['accuracy']))

  if train_eval['accuracy'] > 0.999:
    break
