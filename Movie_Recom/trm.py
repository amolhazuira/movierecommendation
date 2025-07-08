import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
credits = pd.read_csv(r"C:\Users\amolm\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv")
movies = pd.read_csv(r"C:\Users\amolm\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv")
movies = movies.merge(credits,on='title')
movies.isnull().sum()
movies.iloc[0].genres


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
def convert3(obj):
    L =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert3)
movies.head()
movies['crew'][0]
def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()
movies['overview'][0]
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()
new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()
new_df['tags'][0]
new_df['tags'][1]

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors[0]
vectors.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity
new_df[new_df['title'] == 'The Lego Movie'].index[0]
def recommend(movies):
    index = new_df[new_df['title'] == movies].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
recommend('Gandhi')