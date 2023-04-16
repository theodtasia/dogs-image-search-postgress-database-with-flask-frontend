import numpy as np
import PIL
from os import path
from matplotlib import pyplot as plt

from distance_metrics_calculation import DistanceMetricsCalculation, get_images, get_knn_results


def plot_precision_by_metric_for_each_k(precision_dict):
    """
    Plots the precision for each k value and for each distance metric used.
    """
    # Set up the plot
    fig, axs = plt.subplots(len(precision_dict), 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # Plot the precision for each k value and for each distance metric used
    i = 0
    for metric, precision_by_k in precision_dict.items():
        axs[i].plot(list(precision_by_k.keys()), list(precision_by_k.values()))
        axs[i].set_title(f"Precision by k for {metric}")
        axs[i].set_xlabel("k")
        axs[i].set_ylabel("Precision (%)")
        i += 1

    # Save the output
    output_path = f"static/results/precision_by_metric_for_each_k.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    # Close the figure
    plt.close(fig)

def plot_precision_by_metric(precision_dict):
    """
    Plots a bar chart of precision values for each metric for different values of k.
    The input is a dictionary where the keys are the metric names and the values are dictionaries,
    where the keys are the k values and the values are the precision values.
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set the x-axis labels and positions
    nested_keys = set()
    for k in precision_dict:
        nested_keys.update(precision_dict[k].keys())

    metric_names = list(nested_keys)
    x_pos = np.arange(len(metric_names))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)

    # Set the y-axis label and limits
    ax.set_ylabel('Precision (%)')
    ax.set_ylim([0, 100])

    # Set the bar width
    bar_width = 0.25

    # Plot the bars for each k value
    k_values = [5, 10, 20]
    for i, k in enumerate(k_values):
        k_pos = x_pos - bar_width + (i * bar_width) + (bar_width / 2)
        k_precisions = [precision_dict[k][metric] for metric in metric_names]
        ax.bar(k_pos, k_precisions, bar_width, label=f'k={k}')

    # Add the legend
    ax.legend()

    # Add a title and save the plot
    ax.set_title('Precision by Metric and K')
    plt.savefig('static/results/precision_by_metric.png')

    # Close the figure
    plt.close(fig)



def plot_results(filenames, output_name, precision):
    """
    Plots a grid of images specified by filenames and saves the output as output_name.png.
    The maximum number of images that can be plotted is 20.
    """
    # Check that the number of images is <= 20
    num_images = len(filenames)
    if num_images > 20:
        print("Error: maximum number of images is 20")
        return

    # Set up the grid of images
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    num_plots = rows * cols
    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    axs = axs.flatten()

    # Plot each image
    for i in range(num_plots):
        if i < num_images:
            img = plt.imread(filenames[i])
            axs[i].imshow(img)
            axs[i].axis('off')
        else:
            axs[i].set_visible(False)

    output_name_split = output_name.split('_', 2)
    # Add the title with the file name and precision
    title = f"For {output_name_split} - Precision: {precision} %"
    fig.suptitle(title)

    # Save the output
    output_path = f"static/results/{output_name}.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    # Close the figure
    plt.close(fig)


def get_precision(df, image, k, selected_metric, breed, verbose=False):
    selected_image = DistanceMetricsCalculation(PIL.Image.open(image), image)
    knn_results = get_knn_results(selected_image, df, k, selected_metric, verbose)
    correct_breed_count = (knn_results['breed'].values == breed).sum()
    breed_precision = correct_breed_count / len(knn_results.index) * 100
    if verbose:
        print(f'Precision is: {breed_precision}')

    knn_images = knn_results['filename'].apply(lambda x: path.join('static', 'dog_images', x)).tolist()
    plot_results(knn_images,
                 f'''{selected_image.filename.split('.')[0]}_{selected_metric}_k{k}''', breed_precision)
    return knn_results, breed_precision


def evaluate_results():
    image = 'beagle.jpg'
    print(f'For image {image}')
    all_metrics = ['euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard']
    precision_dict = {}
    for k in [5, 10, 20]:
        print(f'With k={k}:')
        breed = 'beagle'
        metric_precision = {}
        for metric in all_metrics:
            knn_results, precision = get_precision(get_images(), path.join('static', 'unknown_images', image), k,
                                                   metric, breed, metric)
            print(f'For metric {metric} precision is {precision:.1f} %')
            metric_precision[metric] = precision
        precision_dict[k] = metric_precision

    # find the best method with the maximum precision
    best_k, best_metric = max(((k, metric) for k, v in precision_dict.items() for metric, precision in v.items()),
                              key=lambda x: x[1])

    print(
        f'The best method is with k={best_k} and metric={best_metric} with a precision of {precision_dict[best_k][best_metric]:.1f}%')
    # plot_precision_by_metric_for_each_k(precision_dict)
    plot_precision_by_metric(precision_dict)


if __name__ == '__main__':
    evaluate_results()
