import math
import numpy as np
import pandas as pd
import requests
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

genius_TOKEN = 'sD0C3epnJdfOQQK4eIC45dHl-Qv7DipToGpuj1n4WeuG5_LDP1HKn31w5Cn1lOux'
genius = lyricsgenius.Genius(genius_TOKEN)
genius.verbose = False
genius.remove_section_headers = True
omdb_TOKEN = 'ce887dbd'
analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer()

centroids = [
[0.06711672, 0.04812787], [0.06227717, 0.21820184], [0.11817479, 0.11362219],
[0.18591696, 0.05011211], [0.05184902, 0.12665329]
]

pth = 'app/irsystem/merged_kmeans.csv'
movies = pd.read_csv(pth)


def get_data(artist, song, movie):
    output = []
    music_result = find_music(artist, song)
    pos = neg = ''
    if song != '':
        output.append('Song: '+music_result.title)
        output.append('Artist: '+music_result.artist)
        output.append('----------------')
        #output.append('Lyrics: '+result.lyrics)
        #output.append('----------------')
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
    else:
        output.append('Artist: '+music_result.name)
        output.append('----------------')
        output.append('Top 3 Songs for this artist:')
        i = 1
        pos = neg = 0
        for x in music_result.songs:
            output.append(str(i) + '. ' + x.title)
            sentiment = analyzer.polarity_scores(x.lyrics)
            pos += sentiment['pos']
            neg += sentiment['neg']
            i += 1
        pos = pos/3
        neg = neg/3
        output.append('----------------')
    label = closest_centroid(pos, neg)
    filtered_movies = get_movie_cluster(label)
    #print(filtered_movies.head(3))
    #movie_to_id = create_mov_to_id(filtered_movies)
    #id_to_index = create_id_to_ind(filtered_movies)
    movie_result = find_movie(movie)
    output.append('Plot of ' + movie_result[0] + ': ' +str(movie_result[1]))
    output.append('----------------')
    return output


def find_music(artist, song=''):
    if song != '':
        result = genius.search_song(song, artist)
    else:
        result = genius.search_artist(artist, max_songs=3)
    return result


def closest_centroid(pos, neg):
    dist = []
    for c in centroids:
        dist.append(math.sqrt((pos - c[0])**2 + (neg - c[1])**2))
    index = dist.index(min(dist))
    return index


def get_movie_cluster(label):
    has_label = movies['k=5']==label
    filtered_movies = movies[has_label]
    return filtered_movies


def cleanjson(result):
    title = result[result.find("Title")+8:result.find("Year")-3]
    plot = result[result.find("Plot")+7:result.find("Language")-3]
    review_imdb = float(result[result.find(
        '"Internet Movie Database","Value":"')+35:result.find('Source":"Rotten Tomatoes"')-8])
    review_rotten = float(result[result.find(
        'Source":"Rotten Tomatoes","Value":')+35: result.find('},{"Source":"Metacritic"')-2])
    return [title, plot, review_imdb, review_rotten]


def response(result):
    text = result[result.find('"Response":')+12:]
    return text.find('True') > -1


def find_movie(movie):
    #output = []
    title = "We could not find your movie"
    plot = ""
    query = "http://www.omdbapi.com/?apikey=" + omdb_TOKEN + "&t=" + movie
    params = {"r": "json", "plot": "full"}
    result = requests.get(query, params)
    if response(result.text):
        json = cleanjson(result.text)
        plot = json[1]
        title = json[0]
        review_imdb = json[2]
        review_rotten = json[3]
        #output.append('Plot of ' + movie + ':' +str(plot))
    else:
        '''output.append(
            "We did not find the movie you searched for. Did you spell it correctly?")
    output.append('----------------')'''
    return (title, plot)


def get_sim(mov1, mov2):
    sum1 = get_summary(mov1)
    sum2 = get_summary(mov2)
    doc = [sum1, sum2]
    tfidf = vectorizer.fit_transform(doc)
    return cosine_similarity(tfidf)[0][1]

def create_mov_to_id(data):
    movie_to_id = {}
    for t, title in enumerate(data['Title']):
        wiki = data['WikiID'][t]
        movie_to_id[title] = wiki
    return movie_to_id

def create_id_to_ind(data):
    id_to_index = {}
    for i, wikiid in enumerate(data['WikiID']):
        id_to_index[wikiid] = i
    return id_to_index

def get_genres(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    genre = movie_data['Genres'][ind]
    return genre

def get_summary(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    summary = movie_data['Summary'][ind]
    return summary

def get_pos(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    pos = movie_data['pos'][ind]
    return pos

def get_neg(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    neg = movie_data['neg'][ind]
    return neg

def get_compound(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    comp = movie_data['compound'][ind]
    return comp

def get_neu(movie):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    neu = movie_data['neu'][ind]
    return neu
