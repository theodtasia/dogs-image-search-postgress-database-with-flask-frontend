from os import path
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard
from imageDBHandler import ImageDBHandler
from feature_extraction import GetImage


class DistanceMetricsCalculation:
    """This class calculates distance metrics between an input image and other images in the dataframe.
It initializes with a pil_image object and the filename."""

    def __init__(self, pil_image, filename) -> None:
        self.image_in_pil_format = pil_image
        self.descriptor_vector = GetImage.get_descriptor_vector(self,
                                                                self.image_in_pil_format)  # get descriptor vector
        # using GetImage class method
        self.filename = path.split(filename)[1]  # extract the filename from the file path

    # Method to return a dictionary of descriptor vector and filename.
    def to_dict(self):
        return {'descriptor_vector': self.descriptor_vector,
                'filename': self.filename}


def get_knn_results(image, df, k, distance_metric='euclidean'):
    # Determine the distance metric to use based on input parameter
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
        # Raise an error if distance metric is not recognized
        raise ValueError(f'Not Found {distance_metric}')

    # Add a 'distance' column to the input dataframe, which contains the distance between each row's descriptor vector
    # and the input image's descriptor vector using the selected distance metric
    df['distance'] = df['descriptor_vector'].apply(
        lambda x: selected_distance_metric(image.descriptor_vector, x))

    # Select the k rows with the smallest distance values, i.e. the k most similar images to the input image
    df_of_k_most_similar_dog_images = df.nsmallest(k, 'distance')

    # Return the dataframe containing the k most similar images
    return df_of_k_most_similar_dog_images


def get_images():
    # Retrieve all images from the image database
    image_db_handler = ImageDBHandler()
    return image_db_handler.get_images()
