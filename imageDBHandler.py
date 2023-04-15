import argparse
import os
import pickle
import shutil
from dotenv import load_dotenv
from os import getenv
from sqlalchemy import create_engine, MetaData, Table, Column, insert
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, BYTEA
from pandas import read_sql
from tqdm import tqdm
from feature_generator import GetImage

class ImageDBHandler:
    """This is a class that manages the connection with the PostgreSQL database.
    Its purpose is to provide various methods to store image static into the database and retrieve it later on as pandas dataframes.
    """

    def __init__(self) -> None:
        self.engine = create_engine(self._get_connection(), pool_pre_ping=True)
        self.target_table = self.create_table()

    def _get_connection(self):
        load_dotenv()
        host = getenv('DB_HOST')
        port = getenv('DB_PORT')
        db = getenv('DB_NAME')
        user = getenv('DB_USER')
        secret = getenv('DB_SECRET')
        return f'postgresql://{user}:{secret}@{host}:{port}/{db}'

    def create_table(self):
        print(f'Add records in dog_images table')
        metadata = MetaData(self.engine)
        table = Table('dog_images',
                      metadata,
                      Column('id', INTEGER, primary_key=True),
                      Column('feature_vector', BYTEA),
                      Column('breed', VARCHAR),
                      Column('filename', VARCHAR),
                      )

        try:
            metadata.create_all(bind=self.engine, checkfirst=True, tables=[table])
        except Exception as e:
            raise e

        return table

    def add_images(self):
        print(f'Adding records in dog_images table')
        dog_images = get_total_files()
        # Filter out existing files
        dog_images_filtered = []
        for image in dog_images:
                dog_images_filtered.append(image)

        insert_statement = insert(self.target_table).values(dog_images_filtered)

        try:
            print(f'Adding {len(dog_images_filtered)} d_images table')
            self.engine.execute(insert_statement)
        except Exception as e:
            raise e
        print('Adding images successfully')

    def delete_images(self):
        delete_statement = self.target_table.delete()
        try:
            self.engine.execute(delete_statement)
        except Exception as e:
            raise e
        print('Deleted all records of dog_images table successfully')

    def get_images(self):
        try:
            query = 'SELECT * FROM dog_images'
            df = read_sql(query, self.engine)

        except Exception as e:
            raise e
        df['feature_vector'] = df['feature_vector'].apply(lambda x: pickle.loads(x))

        return df


def modify_filenames():
    # Set the path to the directory containing the folders
    folder_path = 'static/dog_images'

    # Get a list of all the folder names in the directory
    folder_names = os.listdir(folder_path)

    # Loop through each folder and modify the file names
    for folder_name in folder_names:
        folder_pathname = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_pathname):
            # Get the second string from the folder name
            folder_name_parts = folder_name.split("-")
            if len(folder_name_parts) >= 2:
                folder_name_suffix = folder_name_parts[1]
                # Loop through each file in the folder and modify its name
                for file_name in os.listdir(folder_pathname):
                    file_pathname = os.path.join(folder_pathname, file_name)
                    if os.path.isfile(file_pathname):
                        # Check if the file name already includes the folder name suffix
                        if f"_{folder_name_suffix}." not in file_name:
                            # Add the folder name suffix to the file name
                            new_file_name = f"{file_name.split('.')[0]}_{folder_name_suffix}.{file_name.split('.')[-1]}"
                            new_file_pathname = os.path.join(folder_pathname, new_file_name)
                            os.rename(file_pathname, new_file_pathname)

def extract_files_and_remove_folders(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get all files in folder and its subfolders
    files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    # Extract files to parent folder
    for file in files:
        new_file_path = os.path.join(folder_path, os.path.basename(file))
        if not os.path.exists(new_file_path):
          shutil.move(file, folder_path)

    # Remove folders
    for dirpath, dirnames, filenames in os.walk(folder_path, topdown=False):
        for dirname in dirnames:
            os.rmdir(os.path.join(dirpath, dirname))

    print(f"All files in folder '{folder_path}' have been extracted and folders have been removed.")

def get_dog_images():
    folder_path = 'static/dog_images/'
    modify_filenames()
    extract_files_and_remove_folders(folder_path)
    folder_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            folder_files.append(os.path.join(root, file))

    return folder_files

def get_total_files():
    dog_images = []
    dog_image_files = get_dog_images()
    print('Start processing batch of images')
    for image_file in tqdm(dog_image_files):
        try:
            dog_image = GetImage(image_file)

            if dog_image.feature_vector is not None:
                dog_image.encode_feature_vector()

                dog_images.append(dog_image.to_dict())  # append to dog_images

        except FileNotFoundError as e:
            print(f"Error: {e}. Skipping {image_file}")
            continue

    return dog_images

def get_arguments():
    parser = argparse.ArgumentParser(
        description='handle CLI',
        add_help=True)
    parser.add_argument(
        '--do',
        help='''Do create or add or delete records from the db''',
        type=str,
        choices=['create', 'add', 'delete'],
        required=True)
    return parser.parse_args()


def main():
    args = get_arguments()
    db_handler = ImageDBHandler()
    if args.do == 'create':
        db_handler.create_table()
    elif args.do == 'add':
        db_handler.add_images()
    elif args.do == 'delete':
        db_handler.delete_images()


if __name__ == '__main__':
    main()
