from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and process data once
movies_data = pd.read_csv('final_data.csv')

# Combine features if not already present
def combine_features(row):
    return f"{row['title']} {row['genres']}"

movies_data['combined_features'] = movies_data.apply(combine_features, axis=1)

# Create similarity matrix
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])
similarity = cosine_similarity(feature_vectors)

# Home route
@app.route('/')
def home():
    movie_list = movies_data['title'].tolist()
    return render_template('home.html', movie_list=movie_list)

# Recommend route
@app.route('/Predict', methods=['POST'])
def predict():
    movie_name = request.form['movie']
    count = int(request.form['count'])

    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return render_template('home.html', movie_list=list_of_all_titles, message="Movie not found!")

    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:count+1]):
        index = movie[0]
        title = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title)

    return render_template('home.html', movie_list=list_of_all_titles, recommendations=recommended_movies, selected=movie_name)

if __name__ == '__main__':
    app.run(debug=True)
