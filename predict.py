from solution import num_classes
from solution import test
from solution import train
from solution import ds_config
from solution import model
import matplotlib.pyplot as plt
import math

def plot_images(data, num_classes, cls_pred=None):
  images = data.images()
  cls_true = data.labels()
  grid_size = math.ceil(math.sqrt(data.size))

  rows = grid_size
  cols = math.ceil(float(data.size) / grid_size)

  # Create figure with 3x3 sub-plots.
  fig, axes = plt.subplots(rows, cols)
  fig.subplots_adjust(hspace=1, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    if i >= data.size:
      ax.axis('off')
      continue

    # Plot image.
    ax.imshow(
        images[i],
        cmap='binary')

    # Show true and predicted classes.
    if cls_pred:
      xlabel = "T: {0}, P: {1}".format(cls_true[i], cls_pred[i])
      if cls_true[i] == cls_pred[i]:
        color = "green"
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

  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show()


predictions = list(model.predict(test.predict_input_fn()))
plot_images(train.sorted(), num_classes)
plot_images(test.sorted(), num_classes, predictions)
