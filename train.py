from solution import data
from solution import data_set
from solution import model
from solution import test
from solution import train_expanded

print("Data shape", data.images().shape)
print("Num classes", data_set.num_classes)
print("Train size", train_expanded.size)
print("Test size", test.size)

while True:
  model.train(train_expanded.input_fn(), steps=100)

  train_eval = model.evaluate(train_expanded.input_fn())
  print("Train accuracy", train_eval['accuracy'])

  test_eval = model.evaluate(test.input_fn())
  print("Test accuracy", test_eval['accuracy'])

  if train_eval['accuracy'] > 0.95:
    break
