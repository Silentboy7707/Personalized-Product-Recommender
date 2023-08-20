# Fashion Outfit Recommender

Welcome to the Fashion Outfit Recommender project! This project utilizes a ResNet 50 Model to recommend fashion outfits based on various techniques including Intent Classification, Web Scraping, and Convolutional Neural Networks.

## Project Overview

The project is divided into several sub-models to provide a comprehensive understanding of its functionality.

## Demo Of the Project



https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/1974fdd0-7635-4137-bd9c-08c901bbccbb




### Block Diagram
![WhatsApp Image 2023-08-20 at 18 10 43](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/dffa417e-fa03-49e9-83c1-cec624a144cd)



## Intent Classification

Intent classification involves understanding the user's intent from text data.

1. *Data Collection & Preprocessing* (`intent_preprocess.py`): Raw text data is preprocessed by converting it to lowercase, removing punctuation, and eliminating stopwords using NLTK's built-in stopwords list. Preprocessed data is saved in `preprocessed_intent_data.csv`.

2. *Feature Extraction using TF-IDF* (`intent_featureExtraction.py`): Numerical features are created from preprocessed sentences using the TF-IDF vectorizer. The TF-IDF matrix is saved in `tfidf_matrix.pkl`.

3. *Training and Evaluation of the Classifier* (`train_testclassifier.py`): The dataset is split into training and testing sets. An SVM classifier is initialized, trained on the training data, evaluated on the test data, and the trained model is saved.

## Web Scraping

Web scraping is employed to gather the latest fashion trends.

1. *Scraping Latest Trends*: Utilizing the Pinscrape library, information from platforms like Pinterest, Instagram, and Twitter is scraped. The focus is on Pinterest, and the library is configured to scrape 1 to 3 pages of data.

2. *Downloading and Analyzing Images*: Image details are downloaded after scraping. The scraper retrieves 1 to 3 image URLs based on relevance. Images are cross-checked using labels to ensure personalization.

3. *Personalized Recommendations*: By extracting the latest trends and providing personalized recommendations based on images, users are kept updated on trends tailored to their preferences.

## Convolutional Neural Network (CNN)

In this project, we propose a model that uses Convolutional Neural Network and the Nearest neighbor-backed recommender. As shown in the figure Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in the inventory. The nearest neighbor’s algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/2d64eecc5eec75f86d67bf15d59d87598b7f1a90/Demo/work-model.png?raw=true "Face-Recognition-Attendance-System")

## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/72528f2b4197cc5010227068ec72cd10f71214d4/Demo/resnet.png?raw=true "Face-Recognition-Attendance-System")


## Getting Data

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations.

## Recommendation generation

To generate recommendations, our proposed approach uses Sklearn Nearest neighbors Oh Yeah. This allows us to find the nearest neighbors for the given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.

## Installation
The steps to run:
1.	git clone the repo and `cd` in the folder
2.	Extract the zip files into a folder and open that folder in VS Code. You will then see these files within your folder
    ![ss1](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/8081972c-7f6b-4f12-bbcf-386b228c1d96)
3.	Download the dataset from the below link:
https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
or https://www.kaggle.com/paramaggarwal/fashion-product-images-small
Extract images folder and styles.csv file in our folder. Then following files should then displayed
  ![ss2](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/67daa268-d544-4a59-b6d4-aaa5699a15a9)
4.	Run intent_preprocess.py file, Upon successful execution, you will observe the creation of preprocessed_intent_data.csv file.
5.  Now run intent_featureExtraction.py, you will observe the creation of tlidf_matrix.pkl file after successful completion.
6.	Execute intern_train_testClassifier.py to train your model and intent_classifier_model.pkl will be generated.
![ss5](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/243bf117-11f3-42d6-b515-d61f1c217742)
7.	Execute model_app.py and upon successful execution, embeddings.pkl and filenames.pkl will be generated.
![ss6](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/a831e7f1-9756-4b5e-8ca3-7c4e276dc63d)
8.	Test the model that finds similar products by running model_test.py. A sample image is already provided in the sample folder which will yield similar products.
   
9.	Enter the following command: ‘streamlit run main.py’ to execute our project.
![ss7](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/4fdfd485-29d3-49a5-b3dc-d95b5aab7877)

10.	A new browser window will open. Enter your age, gender (male/female), provide information about type of clothing/occasion you are looking for. The results will be displayed.
![ss8](https://github.com/Silentboy7707/Personalized-Product-Recommender/assets/97505764/af1dce81-cf1d-45d1-8625-d3624f19c995)
NOTE: use pip command to import the libraries


## Acknowledgments

This project was made possible with the support of [Pinscrape](https://github.com/rmcgibbo/pinscrape) for web scraping and the [ResNet50 model](https://keras.io/api/applications/resnet/#resnet50-function) for Convolutional Neural Networks.

Feel free to contribute, report issues, or provide suggestions to enhance this project!
