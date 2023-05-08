import numpy as np
import pickle
from tqdm import tqdm
from os import path
from matplotlib import pyplot as plt
from PIL import Image
from keras.utils import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import random

# Instantiate a pre-trained VGG16 model and create a new model with the same input, but with only the output of the
# fc1 layer

# from keras.applications.resnet50 import preprocess_input, decode_predictions
# base_model = ResNet50(weights='imagenet')

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


class GetImage:
    """Define a class to represent an image and its associated properties (breed, descriptor vector, etc.)"""

    def __init__(self, filename) -> None:
        # Open the image file and store it in PIL format
        self.image_in_pil_format = Image.open(filename)
        # Generate a descriptor vector for the image
        self.descriptor_vector = self.get_descriptor_vector(self.image_in_pil_format)
        # Extract the breed from the image file name
        self.breed = self.get_pet_breed(filename)
        # Store only the file name (without the path) of the image
        self.filename = path.split(filename)[1]
        # Store the path to the folder containing the image
        self.path_to_images_folder = path.split(filename)[0]

    # Decode the descriptor vector, which is stored in a binary format
    def decode_descriptor_vector(self):
        self.descriptor_vector = pickle.loads(self.descriptor_vector)

    # Encode the descriptor vector into a binary format
    def encode_descriptor_vector(self):
        self.descriptor_vector = pickle.dumps(self.descriptor_vector)

    # Get the breed of the pet in the image from the filename
    def get_pet_breed(self, image_filename):
        breed_name = image_filename.split('_')[3]
        return breed_name.replace('.jpg', '')

    # Generate a descriptor vector for the image using the VGG16 model
    def get_descriptor_vector(self, image):
        try:
            # Resize the image to 224x224 and convert it to RGB format
            image = image.resize((224, 224))
            image = image.convert('RGB')
            # Convert the image to a 3D numpy array
            x = img_to_array(image)
            # Add an extra dimension to the array
            x = np.expand_dims(x, axis=0)
            # Preprocess the array to make it compatible with the VGG16 model
            x = preprocess_input(x)
            # Pass the array through the VGG16 model and extract the output of the fc1 layer
            feature = model.predict(x)[0]
        except Exception as e:
            print(e)
            return None
        # Normalize the feature vector
        return feature / np.linalg.norm(feature)

    # Convert the image object to a dictionary
    def to_dict(self):
        return {'descriptor_vector': self.descriptor_vector,
                'breed': self.breed,
                'filename': self.filename}

    # Display the image
    def show_image(self):
        plt.imshow(self.image_in_pil_format)
        plt.show()


# Get a list of randomly selected dog image files
def get_dog_images():
    folder_path = 'static/dog_images'
    random_files = random.sample(folder_path, 500)
    return random_files


# Generate a random sample of dog images and their associated properties
def get_random_sample():
    dog_images = []
    dog_image_files = get_dog_images()
    for image_file in tqdm(dog_image_files):
        dog_image = GetImage(image_file)
        if dog_image.descriptor_vector is not None:
            dog_image.encode_descriptor_vector()
            dog_images.append(dog_image.to_dict())
    return dog_images


if __name__ == '__main__':
    dog_image_files = get_random_sample()
    print(f'Processed {len(dog_image_files)} dog images')
