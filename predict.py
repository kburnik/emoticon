"""
Loads the bootstrapped solution and runs the prediction test with visualization.
"""

from solution import num_classes
from solution import test
from solution import train
from solution import ds_config
from solution import model
import matplotlib.pyplot as plt
import math

def plot_images(data, num_classes, cls_pred=None):
  """Draws a plot with images. Optionally can display prediction results too."""
  images = data.images()
  cls_true = data.labels()
  grid_size = math.ceil(math.sqrt(data.size))

  rows = grid_size
  cols = math.ceil(float(data.size) / grid_size)

  fig, axes = plt.subplots(rows, cols)
  fig.subplots_adjust(hspace=1, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    if i >= data.size:
      ax.axis('off')
      continue

    ax.imshow(
        images[i],
        cmap='binary')

    # Show true and predicted classes.
    if cls_pred:
      xlabel = "T: {0}, P: {1}".format(cls_true[i], cls_pred[i])
      if cls_true[i] == cls_pred[i]:
        color = "green"
        xlabel = "OK"
      else:
        color = "red"
    else:
      xlabel = str(cls_true[i])
      color = "gray"

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_color(color)

    # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()

sorted_train = train.sorted()
sorted_test = test.sorted()

plot_images(
    sorted_train.sorted(),
    num_classes,
    list(model.predict(sorted_train.input_fn())))

plot_images(
    sorted_test,
    num_classes,
    list(model.predict(sorted_test.input_fn())))
