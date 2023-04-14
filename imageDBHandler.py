import argparse
import pickle
from dotenv import load_dotenv
from os import getenv
from sqlalchemy import create_engine, MetaData, Table, Column, insert
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, BYTEA
from pandas import read_sql


class ImageDBHandler:
    """This is a class that manages the connection with the PostgreSQL database.
    Its purpose is to provide various methods to store image data into the database and retrieve it later on as pandas dataframes.
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
        metadata = MetaData(self.engine)

        table = Table('pet_images',
                             metadata,
                             Column('id', INTEGER, primary_key=True),
                             Column('feature_vector', BYTEA),
                             Column('breed', VARCHAR),
                             Column('filename', VARCHAR)
                             )
        try:
            metadata.create_all(bind=self.engine, checkfirst=True, tables=[table])
        except Exception as e:
            raise e

        return table

    def add_images(self):
        from feature_generator import create_pet_images_batch

        dog_images = create_pet_images_batch()

        insert_statement = insert(self.target_table).values(dog_images)

        try:
            print(f'Adding {len(dog_images)} image records in pet_images table')
            self.engine.execute(insert_statement)
        except Exception as e:
            raise e

        print('Added pet images to table successfully')

    def delete_images(self):
        delete_statement = self.target_table.delete()

        try:
            self.engine.execute(delete_statement)
        except Exception as e:
            raise e

        print('Deleted all records of pet_images table successfully')

    def get_images(self):
        try:
            # retrieve pet_images table
            query = 'SELECT * FROM pet_images'
            df = read_sql(query, self.engine)

        except Exception as e:
            raise e

        # decode feature vector
        df['feature_vector'] = df['feature_vector'].apply(lambda x: pickle.loads(x))

        return df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='CLI tool to handle content-based image search database',
        add_help=True)

    parser.add_argument(
        '--action',
        help='''Specify the action to perform: create, add or delete records from the pet_images table in the PostgreSQL database''',
        type=str,
        choices=['create', 'add', 'delete'],
        required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()

    db_handler = ImageDBHandler()

    if args.action == 'create':
        db_handler.create_images_table()
    elif args.action == 'add':
        db_handler.add_images()
    elif args.action == 'delete':
        db_handler.delete_images()


if __name__ == '__main__':
    main()
