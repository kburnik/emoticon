from solution import data
from solution import model
from expander import expander

train, test = data.split(0.7, names=["train", "test"])

train = train.expanded(expander).shuffled()

print("data shape", data.images().shape)
print("train num labels", len(train.labels()))
print("test num labels", len(test.labels()))

for i in range(5):
  model.train(train.input_fn(), steps=1000)
  evaluation = model.evaluate(test.input_fn())
  print("Test accuracy", evaluation['accuracy'])
