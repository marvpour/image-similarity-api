# 🖼 Image Similarity API

Find the 10 most visually similar images for any given image URL

## 📌 Overview

This project provides an API that, given an image URL, returns the 10 most visually similar images from a pre-downloaded dataset. It uses deep learning–based image embeddings to compare images and rank them by similarity.

## 🔥 Example
When searching for this white sneakers:
<img src="images/image_for_search.jpg" alt="Original image" width="50%" />
Here are the first 3 results:
<img src="images/image_1.png" alt="1st results" width="50%" /> <img src="images/image_2.png" alt="2nd results" width="50%" /> <img src="images/image_3.png" alt="3rd results" width="50%" />


## 🚀 Features
- **Image Similarity Search** – Returns top 10 visually similar images for a given input image.
- **Pre-Indexed Dataset** – Images are preprocessed and embedded for faster retrieval.
- **REST API** – Accepts a POST request with an image URL, returns JSON list of similar image URLs.
- **Docker & Conda Support** – Easily run locally or in a containerized environment.

## 🛠 How It Works
1- On the first run, downloads and processes the entire image dataset (takes ~2 hours).
2- Extracts embeddings from each image using a deep learning model (e.g., ResNet, EfficientNet).
3- Stores embeddings for fast similarity search.
4- On request, computes similarity scores and returns the 10 closest matches.

## 📦 Installation

### Option 1 – Conda
1- Install Anaconda or Miniconda. [install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2- Create the environment:
 ```
 conda env create -f environment.yml
 conda activate image-similarity
 ```
3- Start the server:
 ```
 python app.py
```
### Option 2 – Docker
```
docker build -t image-similarity:1.0.0 .
docker run -p 5000:5000 image-similarity:1.0.0
```

## ▶️ Usage
Send a POST request with a JSON body containing an `image_url`:
```
curl -X POST -H "Content-Type: application/json" \
-d '{"url": "https://example.com/myimage.jpg"}' \
http://localhost:5000/similar
```
Response:
```
{
  "similar_images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    ...
    "https://example.com/image10.jpg",
  ]
}
```

## 📚 Technologies Used
- **Python** – Core programming language.
- **Pandas** – Data handling and processing.
- **Requests** – Downloading images from URLs.
- **OpenCV** – Image resizing, preprocessing, and manipulation.
- **NumP**y – Numerical computation and array manipulation.
- **Pillow (PIL)** – Image validation and loading.
- **Matplotlib** – Visualization of results.
- **Scikit-learn** – Label encoding, preprocessing, and train-test split.
- **TensorFlow & Keras** – Model building and inference:
- **TensorFlow Hub** – EfficientNet (feature extraction).
- **VGG16** – Transfer learning for classification.
- **SciPy** – Cosine distance calculation for similarity.
- **Flask** – Serving the similarity API.
- **TQDM** – Progress tracking during image processing and downloads.
- **Docker** – Containerized deployment.
- **Conda** – Environment management.
