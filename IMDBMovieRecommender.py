# Recommend Movies Based on Similar Plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
from numpy import savetxt
from numpy import loadtxt
from nltk.corpus import stopwords


data = pd.read_csv("wiki_movie_plots_deduped.csv", usecols=['Release Year', 'Title', 'Plot'])
data.drop_duplicates(inplace=True)
data['Title'] = data['Title'].astype(str)
data['Title'] = data['Title'].apply(lambda x: x.strip())
data.head()

imdb_titles = pd.read_csv("IMDb movies.csv", usecols=['title', 'year', 'genre'])
imdb_ratings = pd.read_csv("IMDb ratings.csv", usecols=['weighted_average_vote'])

ratings = pd.DataFrame({'Title': imdb_titles.title,
                        'Release Year': imdb_titles.year,
                        'Rating': imdb_ratings.weighted_average_vote,
                        'Genre': imdb_titles.genre,
                        })
ratings.drop_duplicates(inplace=True)
ratings.head()
data = data.merge(ratings, how="left", on=['Title', 'Release Year'])
data.drop_duplicates(inplace=True)
data = data.dropna()
data = data.reset_index(drop=True)

print(data.head())

data.to_pickle("./data.pkl")
movies_df = pd.read_pickle("./data.pkl")

for index, content in movies_df.iterrows():
    paragraph = content.Plot
    paragraph = paragraph.lower()
    paragraph = re.sub(r'\W', ' ', paragraph)
    movies_df.at[index, 'Plot'] = re.sub(r'\s+', ' ', paragraph)

corpus = movies_df[["Title", "Release Year", "Plot", "Rating", "Genre"]]
corpus.to_pickle("./corpus.pkl")
corpus = pd.read_pickle("./corpus.pkl")

stopwords = set(stopwords.words('english'))
# Given a list of words, remove any that are
# in a list of stop words.


def remove_stop_words(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]


word_freq = {}
for index, content in corpus.iterrows():
    paragraph = content.Plot
    tokens = nltk.word_tokenize(paragraph)
    tokens = remove_stop_words(tokens, stopwords)
    for token in tokens:
        if token not in word_freq.keys():
            word_freq[token] = 1
        else:
            word_freq[token] += 1


def sort_freq_dict(freq_dict):
    aux = [(key, freq_dict[key]) for key in freq_dict]
    aux.sort()
    aux.reverse()
    return aux


word_freq = sort_freq_dict(word_freq)
np.save('word_freq.npy', word_freq)

read_dictionary = np.load('word_freq.npy')
desired_array = [int(numeric_string) for numeric_string in read_dictionary[:20, 0]]
plt.barh(read_dictionary[:20, 1], desired_array, align='center', alpha=0.5, color='red')
plt.gca().invert_yaxis()
# Pick the most frequent words and save it in a list
plt.show()
vector_dim = 100
most_freq = list(read_dictionary[:vector_dim, 1])
movie_vectors = []
for index, content in corpus.iterrows():
    paragraph = content.Plot
    sentence_tokens = nltk.word_tokenize(paragraph)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    movie_vectors.append(sent_vec)

movie_vectors = np.asarray(movie_vectors)

savetxt('movie_vectors.csv', movie_vectors, delimiter=',')
corpus = corpus.drop(['Plot'], axis=1)
movie_vectors = loadtxt('movie_vectors.csv', delimiter=',')


def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))


def top_movie_similarities(movie_data, movie_vecs, movie_name, movie_year, above_year, rating, num_movies):
    spacing = 25

    ind = movie_data.index[(movie_data['Title'] == movie_name) & (movie_data['Release Year'] == movie_year)]
    movie_vec = movie_vecs[ind][0]
    vec_mean = movie_vecs.mean(axis=0)
    centered = movie_vecs - vec_mean
    sims = np.array([cosine_similarity(movie_vec - vec_mean, vec) for vec in centered])

    print(f'FILM{" " * (spacing - 2)}Year    IMDB    Genre    COSINE SIMILARITY')
    print(f'{"-" * (40 + spacing)}')
    for i in sorted(sims)[-2::-1]:
        if movie_data.iloc[np.where(sims == i)[0][0]]["Release Year"] >= above_year and movie_data.iloc[np.where(sims == i)[0][0]]["Rating"] >= rating:
            title = movie_data.iloc[np.where(sims == i)[0][0]].Title
            print(
                f'{(title[:spacing - 3] + "...") if len(title) > spacing else title:{spacing}} {movie_data.iloc[np.where(sims == i)[0][0]]["Release Year"]:>5}  {movie_data.iloc[np.where(sims == i)[0][0]]["Rating"]:>5}  {movie_data.iloc[np.where(sims == i)[0][0]]["Genre"]:>5}     {i:.3f}')
            num_movies -= 1
        if num_movies == 0:
            break


top_movie_similarities(corpus, movie_vectors, 'Interstellar', movie_year=2014, above_year=1930, rating=2.0, num_movies=20)