# Redundancy Detection in SRS Documents using NLP

## Project Overview

This project implements a system for detecting redundancy in Software Requirement Specification (SRS) documents using Natural Language Processing (NLP) techniques. It encompasses both a development phase, focused on model training and artifact generation, and a deployment phase, which utilizes a Flask web application to serve the trained models.

## Development Phase: Model Training and Artifact Generation

The core of the NLP redundancy detection is developed and refined in the Jupyter notebooks, primarily `model.ipynb`. This phase is crucial for creating the necessary artifacts that the Flask application later consumes.

### Key Steps in Development:

1.  **Data Ingestion and Exploration**: Initial loading and understanding of the SRS document datasets.
2.  **Text Preprocessing**: Cleaning and preparing raw text data. This involves tokenization, stop-word removal, and potentially stemming/lemmatization to standardize the text.
3.  **Feature Extraction/Vectorization**: Converting processed text into numerical representations (vectors) that machine learning models can understand. The specific vectorization technique (e.g., TF-IDF, Word2Vec, Sentence Transformers) is implemented here.
4.  **Model Training**: Developing and training the NLP model to identify patterns and relationships within the text that indicate redundancy.
5.  **Artifact Generation**: Saving the trained models and preprocessors as `.pkl` files (e.g., `nfr_data.pkl`, `preprocessor.pkl`) into the `artifacts/` directory. These artifacts are the direct output of the development phase and are essential for the deployment phase.

### Models and Transformers Used in Development:

The development process, primarily orchestrated in `model.ipynb` and implemented through modules in `src/components`, utilizes several key NLP models and transformers:

*   **Text Preprocessor (`preprocessor.pkl`)**: This is a custom or off-the-shelf transformer responsible for cleaning, tokenizing, and normalizing the raw text from SRS documents. It typically includes steps like lowercasing, punctuation removal, stop-word filtering, and potentially stemming or lemmatization. Its output is clean, processed text ready for feature extraction.
*   **Feature Extractor/Vectorizer**: While not explicitly named as a separate `.pkl` file in `artifacts/` (it might be part of `preprocessor.pkl` or `nfr_data.pkl`), a vectorization model is crucial. This could be a TF-IDF Vectorizer, a Word2Vec model, a Sentence Transformer model, or even a BERT-based model, which converts the preprocessed text into dense numerical vectors. These vectors are the input for similarity calculations.
*   **Similarity Model**: This component (often implicitly handled by a similarity metric like Cosine Similarity applied to the feature vectors) is responsible for quantifying the likeness between different requirement statements. The `nfr_data.pkl` might contain pre-computed embeddings or a structure optimized for efficient similarity searches.
*   **Redundancy Detection Logic**: This involves applying a threshold to the similarity scores to determine which pairs or groups of requirements are considered redundant. This logic is developed and refined within `model.ipynb`.

These models and transformers are trained and saved as persistent artifacts (`.pkl` files) during the development phase, allowing the Flask application to load and use them efficiently without retraining.

### Outcomes of Development:

The primary outcome of the development phase is the ability to effectively identify and group redundant requirements within SRS documents. This is achieved through:

*   **Identification of Redundant Requirements**: The system can process a set of requirements and flag those that are semantically similar, indicating potential redundancy.
*   **Grouping of Similar Requirements**: Redundant requirements are grouped together, providing a clear overview of overlapping specifications.
*   **Generation of Artifacts**: The `nfr_data.pkl` and `preprocessor.pkl` files are generated, encapsulating the trained models and preprocessing steps necessary for the Flask application to perform real-time redundancy detection. These artifacts are crucial for the deployment phase.

## Deployment Phase: Flask Web Application

The deployment phase focuses on serving the trained NLP models through a user-friendly Flask web interface.

### Features of the Web Application:

*   **Document Upload**: Users can upload SRS documents via a web form.
*   **Redundancy Detection**: The Flask application loads the pre-trained models from the `artifacts/` directory to process the uploaded document and identify similar/redundant requirements.
*   **Interactive Results**: The application presents the identified redundancies in a clear and organized manner to the user.

## Installation

To set up and run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/technical-beast-7/Redundancy-detection-using-NLP
cd Redundancy-detection-using-NLP
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

*   **Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
*   **macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 5. Download NLTK Data

Some NLP functionalities might require NLTK data. Run the following commands within your activated virtual environment:

```bash
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### 1. Run the Flask Application

Ensure your virtual environment is activated, then run the `application.py` file:

```bash
python application.py
```

The application will typically run on `http://127.0.0.1:5000/` or `http://localhost:5000/`.

### 2. Access the Web Interface

Open your web browser and navigate to the address provided in the terminal (e.g., `http://localhost:5000/`).

### 3. Upload and Process Documents

Follow the instructions on the web page to upload your SRS document and initiate the redundancy detection process.

## Project Structure

```
LICENSE                   # Project license information
README.md                 # This file
application.py            # Main Flask application entry point
artifacts/                # Stores trained NLP models and preprocessors (e.g., .pkl files)
notebooks/                # Jupyter notebooks used for data exploration, model training, and artifact generation. Specifically, `model.ipynb` is crucial for understanding the model development process and generating the `nfr_data.pkl` and `preprocessor.pkl` artifacts.
├── datasets/             # Sample datasets used in notebooks
└── model.ipynb           # Notebook for model training and artifact creation
requirements.txt          # Lists all Python dependencies
setup.py                  # Setup script for packaging the project
src/                      # Contains core application logic and NLP components
├── components/           # Modules for data ingestion, transformation, and model training
├── exception.py          # Custom exception handling
├── logger.py             # Logging configuration
├── pipelines/            # Defines data processing and prediction pipelines
└── utils.py              # Utility functions
templates/                # HTML templates for the web interface
├── form.html             # Form for document upload
└── index.html            # Main landing page
uploads/                  # (Optional) Directory for temporary file uploads (should be in .gitignore)
```

## Contributing

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit a pull request with your improvements.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
