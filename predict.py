from solution import num_classes
from solution import data
from solution import ds_config
from solution import model
import matplotlib.pyplot as plt
import math

def plot_images(data, num_classes, cls_pred):
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
    if cls_true[i] == cls_pred[i]:
      xlabel = "Correct"
      color = "green"
    else:
      xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
      color = "red"

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_color(color)

    # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])

  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show()


predictions = list(model.predict(data.predict_input_fn()))
plot_images(data, num_classes, predictions)
