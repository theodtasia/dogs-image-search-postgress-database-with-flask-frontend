import io
import base64  # Import base64 module to encode image in base64 format
from flask import Flask, render_template, request
import PIL
from distance_metrics_calculation import DistanceMetricsCalculation, get_knn_results
from imageDBHandler import ImageDBHandler

app = Flask(__name__)


def get_images():
    """ Returns a dataframe of the table of dog images """
    image_db_handler = ImageDBHandler()
    return image_db_handler.get_images()


try:
    dog_images_df = get_images()
except:
    print('Could not load table of dogs')


# Define a route to handle the form submission and display results
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        distance_metric = request.form['distance_metric']
        k = int(request.form['k'])

        # Get uploaded image
        image = request.files['image']

        if image:
            if image.content_type.split('/')[0] == 'image':
                # Load PIL image
                pil_image = PIL.Image.open(image)
                selected_image = DistanceMetricsCalculation(pil_image, image.filename)

                # Convert PIL image to base64 for displaying in HTML
                img_io = io.BytesIO()
                pil_image.save(img_io, 'JPEG', quality=70)
                img_io.seek(0)
                img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
                selected_image_url = 'data:image/jpeg;base64,' + img_base64

                # Get the k most similar images and their breed counts
                df_of_k_most_similar_images = get_knn_results(selected_image, dog_images_df, k, distance_metric)
                if not df_of_k_most_similar_images.empty:
                    breed_count = df_of_k_most_similar_images['breed'].value_counts()
                filenames = df_of_k_most_similar_images['filename'].tolist()
                print(df_of_k_most_similar_images)
                # Render the index template with query image, breed counts, and similar images
                return render_template('index.html', selected_image=selected_image_url, filenames=filenames,
                                       breed_count=breed_count, k=k)

    # Render the index template with app description and form
    with open('./static/image_search.md', encoding='utf-8') as app_desc:
        app_description = app_desc.read()

    return render_template('index.html', app_description=app_description)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
