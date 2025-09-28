Fake News Detector
This project is a machine learning application designed to classify news articles as either "Real" or "Fake". It uses a TF-IDF Vectorizer for feature extraction and trains four different classification models (Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest) to compare their performance.

The project includes a web interface built with Flask that allows users to input a news article and see the predictions from all four models.

How It Works
Data Loading & Consolidation: The script loads and combines multiple datasets (Fake.csv, True.csv, bbc_news.csv, fake2.csv).

Text Preprocessing: The text is cleaned by converting to lowercase, removing links, punctuation, and numbers.

Model Training: The cleaned data is vectorized using TF-IDF, and four separate models are trained and saved as .pkl files.

Web Application: A Flask app loads the saved vectorizer and all four models to serve predictions on user-provided text.

Setup and Installation
1. Clone the Repository
git clone [https://github.com/XlordCodes/ml-fnd.git](https://github.com/XlordCodes/ml-fnd.git)
cd ml-fnd

2. Create a Virtual Environment
It's highly recommended to use a virtual environment.

Windows:

python -m venv venv
.\venv\Scripts\activate

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a file named requirements.txt with the content below, then install the packages.

requirements.txt:

Flask
scikit-learn
pandas
numpy
nltk

Install command:

pip install -r requirements.txt

4. Download NLTK Data
The application uses NLTK for text processing. Run the following command in a Python interpreter to download the necessary data:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

5. Download Datasets
This project requires the following CSV files in the root directory. Please download them and place them in the folder.

Fake.csv

True.csv

bbc_news.csv

fake2.csv

How to Run the Project
1. Train the Models
First, you must run the training script. This will process the datasets and create the vectorizer and model .pkl files.

python train_and_save.py

This script will print the performance report for each model after training.

2. Run the Web Application
Once the model files (lr_model.pkl, rf_model.pkl, etc.) are generated, you can start the Flask web server. Note: If you have deleted app.py, you should rename app2.py to app.py.

python app.py

3. View the Application
Open your web browser and navigate to:
http://127.0.0.1:5000

You can now paste news text into the form to get a prediction from all four models.