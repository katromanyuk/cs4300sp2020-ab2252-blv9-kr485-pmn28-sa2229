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

'''[
[0.04401261, 0.04654885],
[0.05372926, 0.12658625],
[0.20556818, 0.05363152],
[0.05482889, 0.2243107 ],
[0.11986076, 0.14063887],
[0.11086006, 0.06478736],
]'''

pth = 'app/irsystem/merged_kmeans.csv'
movies = pd.read_csv(pth)


def get_data(artist, song, movie):
    movie_result = find_movie(movie)
    if movie_result=='ERROR':
        return ['We did not find the movie you searched for. Did you spell it correctly?']
    output = []
    music_result = find_music(artist, song)
    pos = neg = ''
    if song != '':
        output.append('Song: '+music_result.title)
        output.append('Artist: '+music_result.artist)
        output.append('----------------')
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
    filtered_movies = get_movie_cluster(label, movies)
    filtered_movies = filtered_movies.reset_index()
    filtered_movies = filtered_movies.drop(['index','Unnamed: 0'], axis=1)
    #print(filtered_movies.head(3))
    movie_to_id = create_mov_to_id(filtered_movies)
    id_to_index = create_id_to_ind(filtered_movies)
    output.append('Movie: ' + movie_result[0])
    #output.append('Plot: ' + str(movie_result[1]))
    output.append('----------------')
    output.append('Your Movie Recommendations Are:')
    output.append('----------------')
    top10 = sorted_top10(movie_result, filtered_movies, movie_to_id, id_to_index)
    output = output + top10
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


def get_movie_cluster(label, movies):
    has_label = movies['k=5']==label
    filtered_movies = movies[has_label]
    return filtered_movies


def cleanjson(result):
    title = result[result.find("Title")+8:result.find("Year")-3]
    plot = result[result.find("Plot")+7:result.find("Language")-3]
    try:
        review_imdb = float(result[result.find(
        '"Internet Movie Database","Value":"')+35:result.find('Source":"Rotten Tomatoes"')-8])
    except:
        review_imdb = -1
    try:
        review_rotten = float(result[result.find(
        'Source":"Rotten Tomatoes","Value":')+35: result.find('},{"Source":"Metacritic"')-2])
    except:
        review_rotten = -1
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
        return "ERROR"
    return (title, plot)


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

def get_summary(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    summary = data['Summary'][ind]
    return summary

def sorted_top10(movie, data, movie_to_id, id_to_index):
    top10 = get_top10(movie, data, movie_to_id, id_to_index)
    top10 = sorted(top10.items(), key=lambda x: x[1], reverse=True)
    sorted10 = []
    for info in top10:
        wiki = info[0]
        ind = id_to_index.get(wiki)
        title = data['Title'][ind]
        sorted10.append(title)
    return sorted10

def get_top10(movie, data, movie_to_id, id_to_index):
    top_movie = {}
    name = movie[0]
    arr = compare_sim(movie, data, movie_to_id, id_to_index)
    wiki_id1 = -1
    if name in movie_to_id:
        wiki_id1 = movie_to_id.get(name)
    top_indices = np.argpartition(-arr, 11)
    list_of_ind = top_indices[:11]
    for ind in list_of_ind:
        title = data['Title'][ind]
        wiki_id2 = data['WikiID'][ind]
        if title != name:
            top_movie[wiki_id2] = arr[ind]
    return top_movie

def compare_sim(movie, data, movie_to_id, id_to_index):
    name = movie[0]
    plot = movie[1]
    wiki_id1 = movie_to_id.get(name)
    arr = np.zeros(len(data))
    for wiki_id2 in data['WikiID']:
        ind = id_to_index.get(wiki_id2)
        movie2 = data['Title'][ind]
        if wiki_id2 == wiki_id1:
            value = 1
        else:
            plot2 = get_summary(movie2, data, movie_to_id, id_to_index)
            value = get_sim(plot, plot2, data, movie_to_id, id_to_index)
        arr[ind] = value
    return arr

def get_sim(plot1, plot2, data, movie_to_id, id_to_index):
    doc = [plot1, plot2]
    tfidf = vectorizer.fit_transform(doc)
    return cosine_similarity(tfidf)[0][1]


def get_genres(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    genre = data['Genres'][ind]
    return genre

def get_pos(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    pos = data['pos'][ind]
    return pos

def get_neg(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    neg = data['neg'][ind]
    return neg

def get_neu(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    neu = data['neu'][ind]
    return neu

def get_compound(movie, data, movie_to_id, id_to_index):
    wiki_id = movie_to_id.get(movie)
    ind = id_to_index.get(wiki_id)
    comp = data['compound'][ind]
    return comp
