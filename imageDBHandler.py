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
from feature_extraction import GetImage


class ImageDBHandler:
    """This is a class that manages the connection with the PostgreSQL database.
       Its purpose is to provide various methods to store image static into the database and retrieve it later on as pandas
       dataframes."""

    def __init__(self) -> None:
        self.engine = create_engine(self._get_connection(),
                                    pool_pre_ping=True)  # create a new database engine with the connection information
        self.target_table = self.create_table()  # create a new table in the database

    def _get_connection(self):
        load_dotenv()  # load environment variables from a .env file
        host = getenv('DB_HOST')  # get the value of the DB_HOST environment variable
        port = getenv('DB_PORT')  # get the value of the DB_PORT environment variable
        db = getenv('DB_NAME')  # get the value of the DB_NAME environment variable
        user = getenv('DB_USER')  # get the value of the DB_USER environment variable
        secret = getenv('DB_SECRET')  # get the value of the DB_SECRET environment variable
        return f'postgresql://{user}:{secret}@{host}:{port}/{db}'  # construct the connection string

    def create_table(self):
        metadata = MetaData(self.engine)  # create a new metadata object
        table = Table('dog_images',  # create a new table object
                      metadata,
                      Column('id', INTEGER, primary_key=True),
                      Column('descriptor_vector', BYTEA),
                      Column('breed', VARCHAR),
                      Column('filename', VARCHAR),
                      )
        try:
            metadata.create_all(bind=self.engine, checkfirst=True, tables=[table])  # create the table in the database
        except Exception as e:
            raise e

        return table  # return the table object

    def add_images(self):
        print(
            f'Adding records in dog_images table')
        dog_images = get_total_files()  # get a list of dog images
        # Filter out existing files
        dog_images_filtered = []
        for image in dog_images:
            dog_images_filtered.append(image)

        insert_statement = insert(self.target_table).values(
            dog_images_filtered)  # create an insert statement for the table

        try:
            print(
                f'Adding {len(dog_images_filtered)} d_images table')
            self.engine.execute(insert_statement)  # execute the insert statement
        except Exception as e:
            raise e
        print(
            'Adding images successfully')

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
        df['descriptor_vector'] = df['descriptor_vector'].apply(lambda x: pickle.loads(x))

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
    modify_filenames()  # Rename image files to lowercase and replace spaces with underscores
    extract_files_and_remove_folders(folder_path)  # Extract folder files and remove redundant subfolders
    folder_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            folder_files.append(os.path.join(root, file))  # Collect all image file paths

    return folder_files


def get_total_files():
    dog_images = []
    dog_image_files = get_dog_images()  # Get all image file paths
    print('Start processing batch of images')
    for image_file in tqdm(dog_image_files):  # Loop over each image file path
        try:
            dog_image = GetImage(image_file)  # Create a GetImage object

            if dog_image.descriptor_vector is not None:  # Check if the descriptor vector exists
                dog_image.encode_descriptor_vector()  # Compute the descriptor vector

                dog_images.append(dog_image.to_dict())  # Convert the GetImage object to a dictionary and add it to
                # dog_images

        except FileNotFoundError as e:
            print(f"Error: {e}. Skipping {image_file}")  # If an error occurs, print a message and skip to the next
            # image
            continue

    return dog_images


def get_arguments():
    parser = argparse.ArgumentParser(
        description='handle CLI',  # Create an ArgumentParser object to handle command line arguments
        add_help=True)
    parser.add_argument(
        '--do',
        help='''Do create or add or delete records from the db''',  # Add a command line argument to specify the
        # operation to be performed
        type=str,
        choices=['create', 'add', 'delete'],
        required=True)
    return parser.parse_args()  # Parse the command line arguments and return the result


def main():
    args = get_arguments()  # Parse the command line arguments
    db_handler = ImageDBHandler()  # Create a ImageDBHandler object
    if args.do == 'create':  # If the 'create' command is specified, create a new table in the database
        db_handler.create_table()
    elif args.do == 'add':  # If the 'add' command is specified, add image records to the database
        db_handler.add_images()
    elif args.do == 'delete':  # If the 'delete' command is specified, delete image records from the database
        db_handler.delete_images()
