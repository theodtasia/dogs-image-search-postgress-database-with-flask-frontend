from os import path
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard
from imageDBHandler import ImageDBHandler
from feature_extraction import GetImage


class DistanceMetricsCalculation:
    def __init__(self, pil_image, filename) -> None:
        self.image_in_pil_format = pil_image
        self.descriptor_vector = GetImage.get_descriptor_vector(self, self.image_in_pil_format)
        self.filename = path.split(filename)[1]

    def to_dict(self):
        return {'descriptor_vector': self.descriptor_vector,
                'filename': self.filename}


def get_knn_results(image, df, k, distance_metric='euclidean', verbose=False):
    if distance_metric == 'cosine':
        selected_distance_metric = cosine
    elif distance_metric == 'cityblock':
        selected_distance_metric = cityblock
    elif distance_metric == 'euclidean':
        selected_distance_metric = euclidean
    elif distance_metric == 'minkowski':
        selected_distance_metric = minkowski
    elif distance_metric == 'chebyshev':
        selected_distance_metric = chebyshev
    elif distance_metric == 'jaccard':
        selected_distance_metric = jaccard
    elif distance_metric == 'canberra':
        selected_distance_metric = canberra
    else:
        raise ValueError(f'Not Found {distance_metric}')

    df['distance'] = df['descriptor_vector'].apply(
        lambda x: selected_distance_metric(image.descriptor_vector, x))
    df_of_k_most_similar_dog_images = df.nsmallest(k, 'distance')
    if verbose:
        print(f'Top {k} most similar images of {image.filename}')
        for index, item in df_of_k_most_similar_dog_images.iterrows():
            print(' breed {} and image with filename {}'.format(item['breed'], item['filename']))

    return df_of_k_most_similar_dog_images


def get_images():
    image_db_handler = ImageDBHandler()
    return image_db_handler.get_images()
