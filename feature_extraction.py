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

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


class GetImage:

    def __init__(self, filename) -> None:
        self.image_in_pil_format = Image.open(filename)
        self.descriptor_vector = self.get_descriptor_vector(self.image_in_pil_format)
        self.breed = self.get_pet_breed(filename)
        self.filename = path.split(filename)[1]
        self.path_to_images_folder = path.split(filename)[0]

    def decode_descriptor_vector(self):
        self.descriptor_vector = pickle.loads(self.descriptor_vector)

    def encode_descriptor_vector(self):
        self.descriptor_vector = pickle.dumps(self.descriptor_vector)

    def get_pet_breed(self, image_filename):
        breed_name = image_filename.split('_')[3]
        return breed_name.replace('.jpg', '')

    def get_descriptor_vector(self, image):
        try:
            # VGG must take a 224x224 img as an input
            image = image.resize((224, 224))
            image = image.convert('RGB')
            x = img_to_array(image)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)[0]
        except Exception as e:
            print(e)
            return None
        return feature / np.linalg.norm(feature)

    def to_dict(self):
        return {'descriptor_vector': self.descriptor_vector,
                'breed': self.breed,
                'filename': self.filename}

    def show_image(self):
        plt.imshow(self.image_in_pil_format)
        plt.show()


def get_dog_images():
    folder_path = 'static/dog_images'
    random_files = random.sample(folder_path, 500)
    return random_files


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
