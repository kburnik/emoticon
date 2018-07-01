from solution import data_set
from solution import data
from solution import model
from expander import expander

train, test = data.split(0.7, names=["train", "test"])

train = train.expanded(expander).shuffled()

print("Data shape", data.images().shape)
print("Num classes", data_set.num_classes)
print("Train size", train.size)
print("Test size", test.size)

for i in range(5):
  model.train(train.input_fn(), steps=1000)
  evaluation = model.evaluate(test.input_fn())
  print("Test accuracy", evaluation['accuracy'])
