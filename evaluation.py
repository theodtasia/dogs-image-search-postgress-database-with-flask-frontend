import pandas as pd
import PIL
from os import path
from distance_metrics_calculation import DistanceMetricsCalculation, k_most_similar_images, load_database_images


def get_precision(df, image, k, selected_metric, breed, verbose=False):
    selected_image = DistanceMetricsCalculation(PIL.Image.open(image), image)
    knn_results = k_most_similar_images(selected_image, df, k, selected_metric, verbose)
    correct_breed_count = (knn_results['breed'].values == breed).sum()
    breed_precision = correct_breed_count / len(knn_results.index)
    if verbose:
        print(f'Precision is: {breed_precision}')

    # knn_images = knn_results['filename'].apply(lambda x: path.join('static', 'images', x)).tolist()
    # plot_results\(knn_images,
    #                        f'''{selected_image.filename.split('.')[0]}_{selected_metric}_k{k}''')
    return knn_results, breed_precision


def metrics_results(image, k, breed):
    dog_images = load_database_images()
    metrics = pd.DataFrame(None)
    all_metrics = ['euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard']
    for metric in all_metrics:
        knn_results, precision = get_precision(dog_images, image, k, metric, breed)
        print(f'For metric {metric} precision is {precision}')
        metrics = pd.concat([metrics, knn_results])


def evaluate_results():
    image = 'test.jpg'
    print(f'For image {image}')
    for k in [5, 10, 20]:
        print(f'With k={k}:')
        query_pet_image_breed = 'test'
        print(f'For image {image}')
        metrics_results(path.join('static', 'unknown_images', image), k, query_pet_image_breed)


if __name__ == '__main__':
    evaluate_results()
