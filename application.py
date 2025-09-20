from flask import Flask, request, render_template
import os
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from sentence_transformers import SentenceTransformer, util

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))


def process_requirement_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [
        lemmatizer.lemmatize(word) if not word.isdigit() else word
        for word in text if word not in english_stopwords
    ]
    return ' '.join(text)


def read_requirements_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 1)
            if len(parts) == 2:
                data.append(parts)
    return data


@app.route('/', methods=['GET', 'POST'])
def index():
    print("Index function accessed.")
    if request.method == 'POST':
        print("POST request received.")
        files = request.files.getlist('nfr_file')
        if not files or all(f.filename == '' for f in files):
            print("No files selected or uploaded.")
            return render_template('index.html', message="No files selected.")

        all_data = []

        # Ensuring order and appending content sequentially
        for file in files:
            if file and file.filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                print(f"File saved: {filepath}")
                try:
                    file_data = read_requirements_file(filepath)
                    if file_data:
                        all_data += file_data  # Append to maintain sequential order
                        print(f"Data read from {file.filename}: {len(file_data)} entries.")
                except Exception as e:
                    print(f"Error reading file {file.filename}: {str(e)}")
                    return render_template('index.html', message=f"Error in {file.filename}: {str(e)}")

        if not all_data:
            print("No valid data found in uploaded files.")
            return render_template('index.html', message="No valid data found in uploaded files.")

        # Create DataFrame and process
        df = pd.DataFrame(all_data, columns=["class", "requirement"])
        print(f"DataFrame created with {len(df)} entries.")
        df['RequirementText'] = df['requirement'].apply(process_requirement_text)

        try:
            # Generate BERT embeddings - Returns a 2D array
            embeddings = model.encode(df['RequirementText'].tolist(), convert_to_tensor=False)
            print("BERT embeddings generated.")
            # Flatten the embeddings into a list for proper storage
            df['BERTEmbedding'] = embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return render_template('index.html', message=f"Error generating embeddings: {str(e)}")

        # Similarity comparison
        similarity_threshold = 0.8
        df['SimilarRequirement'] = None

        for i in range(len(df)):
            similar_reqs = []
            for j in range(i + 1, len(df)):
                try:
                    sim_score = util.cos_sim(df['BERTEmbedding'][i], df['BERTEmbedding'][j]).item()
                    if sim_score > similarity_threshold:
                        similar_reqs.append(f"{df['requirement'][j]} (ID: {j})")
                except Exception as e:
                    print(f"Error calculating similarity: {str(e)}")
                    return render_template('index.html', message=f"Error calculating similarity: {str(e)}")

            if similar_reqs:
                df.at[i, 'SimilarRequirement'] = ', '.join(similar_reqs)

        similar_count = df['SimilarRequirement'].count()
        result_data = df[['class', 'requirement', 'SimilarRequirement']].to_dict(orient='records')

        # Debugging: Print the resulting DataFrame
        print("Processed DataFrame:", df[['class', 'requirement', 'SimilarRequirement']])
        print(f"Rendering form.html with {len(result_data)} results and {similar_count} similar items.")
        print(f"Result data: {result_data}")
        print(f"Similar count: {similar_count}")
        return render_template('form.html', results=result_data, similar_count=similar_count)

    return render_template('index.html')


if __name__ == '__main__':
    # Download NLTK resources once, if not present
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    print("Starting server at http://127.0.0.1:5050")
    app.run(debug=True, port=5050)
