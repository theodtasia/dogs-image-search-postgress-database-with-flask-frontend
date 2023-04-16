import numpy as np
import pandas as pd
import PIL
from os import path
from matplotlib import pyplot as plt

from distance_metrics_calculation import DistanceMetricsCalculation, get_images, get_knn_results


def plot_precision_by_metric(precision_dict):
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
    output_path = f"static/results/precision_by_metric_{metric}.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

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
    if len(output_name_split) > 2:
        string_after_second_underscore = output_name_split[2]
        print(string_after_second_underscore)
    else:
        print("Error: output name does not contain at least 2 underscores")

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


def metrics_results(image, k, breed):
    dog_images = get_images()
    metrics = pd.DataFrame()
    all_metrics = ['euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard']
    precision_dict = {}
    for metric in all_metrics:
        knn_results, precision = get_precision(dog_images, image, k, metric, breed)
        print(f'For metric {metric} precision is {precision} %')
        precision_dict[metric] = precision
        metrics = pd.concat([metrics, knn_results])
    best_metric = max(precision_dict, key=precision_dict.get)
    print(f'The metric with the highest precision for k = {k} is {best_metric}')
    return metrics


def evaluate_results():
    image = 'beagle.jpg'
    print(f'For image {image}')
    for k in [5, 10, 20]:
        print(f'With k={k}:')
        breed = 'beagle'
        metrics_results(path.join('static', 'unknown_images', image), k, breed)


if __name__ == '__main__':
    evaluate_results()
