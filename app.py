from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load the TF-IDF vectorizer and matrix
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')

# Assuming df is also saved and needs to be loaded
df = joblib.load('df.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'qa_pairs' not in session:
        session['qa_pairs'] = []
    
    if request.method == 'POST':
        question = request.form['question']
        answers = generate_answer(question)
        # Store question and answers as a dictionary
        session['qa_pairs'] = [{'question': correct_spelling(question), 'answers': answers}] + session['qa_pairs']
    
    return render_template('index.html', qa_pairs=session['qa_pairs'])

@app.route('/start', methods=['GET'])
def clear():
    # Clear the session
    session.clear()
    return redirect(url_for('home'))

def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)

def generate_answer(question, top_n=5):
    corrected_question = correct_spelling(question)
    # Vectorize the question using the loaded TF-IDF vectorizer
    question_vector = tfidf_vectorizer.transform([corrected_question])

    # Compute cosine similarity between the question vector and the TF-IDF matrix
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    top_n_indices = similarities.argsort()[-top_n:][::-1]
    top_n_scores = similarities[top_n_indices]

    results = []
    for i in range(top_n):
        score = round(top_n_scores[i], 3)
        # the item is not relevant to the question if score < 0.1
        if score < 0.1:
            continue
        index = top_n_indices[i]
        results.append({
            'site_name': df.iloc[index]['SiteName'],
            'description': df.iloc[index]['Description'],
            'interests': df.iloc[index]['Interests'],
            'score': round(top_n_scores[i], 3)
        })
        if not results:
            return [{"site_name": "N/A", "description": "No relevant results found. Please try a different query.", "interests": "N/A", "score": "N/A"}]
    
    return results 

if __name__ == '__main__':
    app.run(debug=True)
