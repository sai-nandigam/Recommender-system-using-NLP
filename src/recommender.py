import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(df):
    # Handle missing values
    df['description'] = df['description'].fillna('')
    df['genres'] = df['genres'].fillna('')
    
    # Combine text
    df['content'] = df['description'].astype(str) + " " + df['genres'].astype(str)
    
    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Similarity
    similarity = cosine_similarity(tfidf_matrix)
    
    return similarity

def recommend(title, df, similarity):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    if title not in indices:
        raise Exception("Movie not found")

    idx = indices[title]
    
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    
    movie_indices = [i[0] for i in scores]
    
    return df['title'].iloc[movie_indices]
    

