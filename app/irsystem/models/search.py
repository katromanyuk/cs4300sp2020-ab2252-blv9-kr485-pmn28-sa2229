import math
import numpy as np
import pandas as pd
import pickle
import requests
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

genius_TOKEN = 'sD0C3epnJdfOQQK4eIC45dHl-Qv7DipToGpuj1n4WeuG5_LDP1HKn31w5Cn1lOux'
genius = lyricsgenius.Genius(genius_TOKEN)
genius.verbose = False
genius.remove_section_headers = True
omdb_TOKEN = 'ce887dbd'
analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, max_df=0.8, min_df=20, norm='l2')
tokenizer = vectorizer.build_tokenizer()

movies = pd.read_csv('app/irsystem/merged_data.csv')
num_movies = len(movies)
#inv_idx = np.load('app/irsystem/inv_idx.npy',allow_pickle='TRUE').item()
norms = np.loadtxt('app/irsystem/norms.csv', delimiter=',')
with open('app/irsystem/inv_idx.pkl', 'rb') as f:
     inv_idx = pickle.load(f)


def get_data(artist, song, movie):
    output = []
    movie_result = find_movie(movie)
    if movie_result=='ERROR':
        return ['We did not find the movie you searched for. Did you spell it correctly?']
    music_result = find_music(artist, song)
    pos = neg = ''
    if song != '':
        output.append(music_result.title+' by '+music_result.artist)
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
    else:
        output.append('Artist: '+music_result.name)
        output.append('----------------')
        output.append('Top 3 Songs for this artist:')
        i = 1
        pos = neg = neu = 0
        for x in music_result.songs:
            output.append(str(i) + '. ' + x.title)
            sentiment = analyzer.polarity_scores(x.lyrics)
            pos += sentiment['pos']
            neg += sentiment['neg']
            neu += sentiment['neu']
            i += 1
        pos = pos/3
        neg = neg/3
        neu = neu/3
    #movies = listify(movies)
    output.append('----------------')
    pos_p = str(round(pos*100,2))
    neg_p = str(round(neg*100,2))
    neu_p = str(round(neu*100,2))
    s = 'Your music choice is '+pos_p+'% positive, '+neg_p+'% negative, and '\
    +neu_p+'% neutral'
    output.append(s)
    output.append('----------------')
    output.append('Movie: ' + movie_result[0])
    output.append('----------------')
    output.append('Your Movie Recommendations Are:')
    #idf = compute_idf(inv_idx,num_movies)
    #results = index_search(movie_result[1],idf)
    #ten = get_10(movie_result[0],results)
    #output = output + ten
    return output


def find_music(artist, song=''):
    if song != '':
        result = genius.search_song(song, artist)
    else:
        result = genius.search_artist(artist, max_songs=3)
    return result


def find_movie(movie):
    title = ""
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
    else:
        return "ERROR"
    return (title, plot, review_imdb, review_rotten)


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


def listify(df):
    genres = []
    languages = []
    countries = []
    for x,y,z in zip(df['Genres'],df['Languages'],df['Countries']):
        g = re.findall(': \"(.*?)\"', x)
        l = re.findall(': \"(.*?)\"', y)
        c = re.findall(': \"(.*?)\"', z)
        genres.append(g)
        languages.append(l)
        countries.append(c)
    df['Genres'] = genres
    df['Languages'] = languages
    df['Countries'] = countries
    return df


def compute_idf(inv_idx, n_docs, min_df=20, max_df_ratio=0.8):
    idf = {x: math.log2(n_docs/(1+len(inv_idx[x]))) for x in inv_idx
           if len(inv_idx[x])>=min_df and len(inv_idx[x])/n_docs<=max_df_ratio}
    return idf


def index_search(query,idf):
    #movie = find_movie(user_mov)
    #query = movie[1]
    #query = ''
    #if movie[0] in movie_titles:
    #    query = movies['Summary'][title_to_index[movie[0]]]
    #else:
    #    query = movie[1]
    scores = np.zeros(len(norms))
    docs = [i for i in range(len(norms))]
    q = query.lower()
    q_tokens = tokenizer(q)
    q_norm_sq = 0
    for t in set(q_tokens):
        if t in idf:
            q_norm_sq += (q_tokens.count(t)*idf[t])**2
            for (doc,cnt) in inv_idx[t]:
                scores[doc] += (q_tokens.count(t)*cnt*idf[t]**2)/norms[doc]
    q_norm = math.sqrt(q_norm_sq)
    new_scores = [score/q_norm for score in scores]
    pos = [x for x in movies['pos']]
    neg = [x for x in movies['neg']]
    result = sorted(tuple(zip(new_scores,pos,neg,docs)),reverse=True)
    return result[:100]


#def weigh_score(cosim,pos,neg,rating,w1,w2,w3):
    #total = 0
    #dist.append(math.sqrt((pos - c[0])**2 + (neg - c[1])**2))


def get_10(movie,results):
    ten = []
    i = 1
    for (score,p,n,ind) in results[:10]:
        if movies['Title'][ind] != movie:
            ten.append(str(i)+'.')
            ten.append(movies['Title'][ind])
            ten.append('Score: '+str(sim))
            i+=1
    return ten













'''def closest_centroid(pos, neg):
    dist = []
    for c in centroids:
        dist.append(math.sqrt((pos - c[0])**2 + (neg - c[1])**2))
    index = dist.index(min(dist))
    return index

def get_movie_cluster(label, movies):
    has_label = movies['k=5']==label
    filtered_movies = movies[has_label]
    return filtered_movies

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

def get_sim2(plot1, plot2, data, movie_to_id, id_to_index):
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
    return comp'''
