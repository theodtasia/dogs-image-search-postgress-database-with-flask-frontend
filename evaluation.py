import pandas as pd
import PIL
from os import path
from distance_metrics_calculation import DistanceMetricsCalculation, get_images, get_knn_results


def get_precision(df, image, k, selected_metric, breed, verbose=False):
    selected_image = DistanceMetricsCalculation(PIL.Image.open(image), image)
    knn_results = get_knn_results(selected_image, df, k, selected_metric, verbose)
    correct_breed_count = (knn_results['breed'].values == breed).sum()
    breed_precision = correct_breed_count / len(knn_results.index) * 100
    if verbose:
        print(f'Precision is: {breed_precision}')

    # knn_images = knn_results['filename'].apply(lambda x: path.join('static', 'images', x)).tolist()
    # plot_results\(knn_images,
    #                        f'''{selected_image.filename.split('.')[0]}_{selected_metric}_k{k}''')
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
