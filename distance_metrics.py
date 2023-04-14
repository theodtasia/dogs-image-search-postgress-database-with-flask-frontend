from os import path
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard

from imageDBHandler import ImageDBHandler
from feature_generator import PetImage


class DistanceMetrics:
    def __init__(self, pil_image, filename) -> None:
        self.image_in_pil_format = pil_image
        self.feature_vector = PetImage.extract_feature_vector(self, self.image_in_pil_format)
        self.filename = path.split(filename)[1]

    
    def to_dict(self):
        return {'feature_vector': self.feature_vector,
                'filename': self.filename}


def k_most_similar_images(query_image, dog_images_df, k, distance_metric='cosine', verbose=False):

    if distance_metric == 'cosine':
        selected_distance_metric = cosine
    elif distance_metric == 'euclidean':
        selected_distance_metric = euclidean
    elif distance_metric == 'cityblock':
        selected_distance_metric = cityblock
    elif distance_metric == 'minkowski':
        selected_distance_metric = minkowski
    elif distance_metric == 'chebyshev':
        selected_distance_metric = chebyshev
    elif distance_metric == 'canberra':
        selected_distance_metric = canberra
    elif distance_metric == 'jaccard':
        selected_distance_metric = jaccard
    else:
        raise ValueError(f'Not Found {distance_metric}')
    
    dog_images_df['distance'] = dog_images_df['feature_vector'].apply(lambda x: selected_distance_metric(query_image.feature_vector, x))

    df_of_k_most_similar_dog_images = dog_images_df.nsmallest(k, 'distance')

    if verbose:
        print(f'Top {k} most similar images of {query_image.filename}')
        for index, item in df_of_k_most_similar_dog_images.iterrows():
            print(' breed {} and image with filename {}'.format(item['breed'], item['filename']))
    
    return df_of_k_most_similar_dog_images


def load_database_images():
    """ Returns as a dataframe the table of dog images
    """
    image_db_handler = ImageDBHandler()

    return image_db_handler.get_images()