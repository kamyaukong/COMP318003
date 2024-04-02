#
# Data Prep and training for an Open Data of Cultural Hotspot - Points of Interest
# https://open.toronto.ca/dataset/cultural-hotspot-points-of-interest/
#
import pandas as pd
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Define the preprocessing text function
def preprocess_text(text):
    # Check if the text is a string instance; if not (i.e., it's NaN or a number), return an empty string
    if not isinstance(text, str):
        return ''
    
    # Proceed with preprocessing if it's a string
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Word Tokenization - split sentense into word
    tokens = word_tokenize(text)
    # Stop Words - remove stop words such as: is the a are ...
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization - 
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def retrieve_most_relevant_entries(question, vectorizer, tfidf_matrix, df, top_n=5, threshold=0.1):
    question_vector = vectorizer.transform([question])
    
    # Compute cosine similarity between question vector and TF-IDF matrix
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    # Check if all similarities are below the threshold to address unrelated questions
    if max(similarities) < threshold:
        relevant_entries = pd.DataFrame({'SiteName': ['N/A'], 
                                     'Description': ['Your question seems unrelated or too general. Could you please specify or ask something else?'], 
                                     'Interests': ['N/A'], 
                                     'Score': [0]},
                                    index=[0])  # Specify an index here
        return relevant_entries
    
    # Get the top N similar entries
    top_n_indices = similarities.argsort()[-top_n:][::-1]  # Get the indices of the top N similarities

    # Retrieve the top N entries and their scores
    top_n_scores = similarities[top_n_indices]

    # Create a DataFrame with the results and scores
    relevant_entries = df.iloc[top_n_indices].copy()
    # score column is just to review after training
    relevant_entries['Score'] = top_n_scores
    return relevant_entries

# Load dataset into a pandas DataFrame
df = pd.read_csv('points-of-interest-4326.csv')

#
# Column 'descriptoin' is the main data to handle user query
# Column 'features' is a combined feature matrix that includes both text and categorical data
#
# Preprocess the 'Description' column
df['Processed_Description'] = df['Description'].apply(preprocess_text)

# Display the DataFrame
df[['Description', 'Processed_Description']]

# Vectorize the processed text descriptions
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 2), stop_words='english', norm='l2', use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Description'])

# One-hot encode the 'Interests' column
onehot_encoder = OneHotEncoder()
interests_encoded = onehot_encoder.fit_transform(df[['Interests']])

# Combine the TF-IDF features with the one-hot encoded features
features = hstack([tfidf_matrix, interests_encoded])

# Save vectorized data to lib file for server to perform query
joblib.dump(df, 'df.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
