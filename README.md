# **Content-Based Dog Image Retrieval**
This project is a straightforward implementation of a content-based image retrieval engine that utilizes Python for the application logic and PostgreSQL as a storage backend. The VGG16 model is employed to extract features from images. The UI Application was developed using Flask.

# **Dataset**
The Stanford Dog Dataset is a sizable and thorough collection of over 20,000 annotated photos of dog breeds representing 120 varieties.
# **Requirements**
To use this project, you must create a Python virtual environment and install the packages found in the base_requirements.txt file.

To create and activate a Python virtual environment using Anaconda:

```
conda create --name image_search_engine env python=3.8
conda activate image_search_engine
```
To install the necessary base Python packages:

```
pip install -r requirements.txt
```

# **PostgreSQL:**
To use this application, you must establish a connection with a PostgreSQL database and create a table named dog_images. To establish a connection with PostgreSQL, create a .env file in the following format:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=image_search
DB_USER=postgres
DB_SECRET=mpompos
```

# **Setting up the Dataset of Images:**

Go to the following link: http://vision.stanford.edu/aditya86/ImageNetDogs/

Scroll down and click on the "Download images" link.

This will download a file named "images.tar" to your computer.

Extract the contents of the tar file using a file archiver program such as 7-Zip or WinZip. This will create a folder named "Images" containing the dataset.

Add the breed folders inside the dog_images directory of the project

# **Setting up the App:**
The application can then be started by typing the command:
```
python main.py
```
# **Setting up the PostgreSQL Database:**

Run the following command to create the dog_images table:

```
python db.py --do=create
```

Run the following command to add the dog images records into to the table of the PostgreSQL database. 

```
python db.py --do=add
```

# **Evaluation**

if you want to evaluate the results and make plots for an unkown image
update the following consts in evaluation.py with your imageName and breed

 ```  
 image = 'beagle.jpg'
 breed = 'Beagle' 
 ```
 
At the breed const write the name of the dog breed that your image includes
in EXACTLY same way it appears in the stanford image folders, 
for example the correct breed name is pug and not PUG or Pug.
Check the dataset folder names.

Run the evaluation.py