from solution import data_set
from solution import data
from solution import train_expanded
from solution import test
from solution import model

print("Data shape", data.images().shape)
print("Num classes", data_set.num_classes)
print("Train size", train.size)
print("Test size", test.size)

while True:
  model.train(train_expanded.input_fn(), steps=1000)

  train_eval = model.evaluate(train_expanded.input_fn())
  print("Train accuracy", train_eval['accuracy'])

  test_eval = model.evaluate(test.input_fn())
  print("Test accuracy", test_eval['accuracy'])

  if train_eval['accuracy'] > 0.98:
    break
