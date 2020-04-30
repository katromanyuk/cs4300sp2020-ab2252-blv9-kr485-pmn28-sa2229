import csv
import math
import json
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
omdb_TOKEN = '4292bf53'
analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, max_df=0.8, min_df=20, norm='l2')
tokenizer = vectorizer.build_tokenizer()


movies = pd.read_csv('app/merged_data3.csv')
n_mov = len(movies)
norms = np.loadtxt('app/norms.csv', delimiter=',')

inv_idx = np.load('app/inv_idx.npy',allow_pickle='TRUE').item()
idf = {x: math.log2(n_mov/(1+len(inv_idx[x]))) for x in inv_idx if len(inv_idx[x])>=20 and len(inv_idx[x])/n_mov<=0.8}


def get_data(artist, song, movie, quote):
    output = []
    movie_result = find_movie(movie)
    if movie_result=='ERROR':
        return ['We did not find the movie you searched for. Did you spell it correctly?']
    music_result = find_music(artist, song)

    if song != '':
        song_disp = 'Song: ' + music_result.title + ' by ' + music_result.artist
        top3 = get_songs(music_result.artist)
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
        comp = sentiment['compound']
    else:
        song_disp = 'Artist: ' + music_result.name
        top3 = get_songs(music_result.name)
        sent_list = get_artist_sentiment(music_result.name)
        pos = sent_list[0]
        neg = sent_list[1]
        neu = sent_list[2]
        comp = sent_list[3]
        
    if quote != '':
        val = getquote(quote)
        pos += val[0]
        neg += val[1]
        neu += val[2]
        comp += val[3]
        pos /= 2
        neg /= 2
        neu /= 2
        comp /=2
        quote_disp = quote
    else:
        quote_disp = "N/A"

    pos_p = str(round(pos*100,2))
    neg_p = str(round(neg*100,2))
    neu_p = str(round(neu*100,2))
    s1 = 'Music Sentiment: '+pos_p+'% positive, '+neg_p+'% negative, and '+neu_p+'% neutral'
    s2 = 'Compound Sentiment: '+str(round(comp,4))+' ('+sent_type(comp)+')'

    row1 = [    [song_disp],            ['Movie: '+movie_result[0]]   ]
    row2 = [    ['Top 3 Songs: ']+top3, ['Summary: ', movie_result[1]]  ]
    row3 = [    [s1, s2],               ['Quote: ', quote_disp]    ]
    output = [row1, row2, row3]

    dists = get_sent_dist(comp)
    scores = get_scores(movie_result[1], dists)
    ten = print_ten(movie_result[0],scores)
    output = [output] + ten
    return output


def placeholder_function(artist, song, movie, quote):
    output = []
    movie_result = find_movie(movie)
    if movie_result=='ERROR':
        return ['We did not find the movie you searched for. Did you spell it correctly?']
    music_result = find_music(artist, song)

    if song != '' and artist != '':
        output.append('Song: '+music_result.title+' by '+music_result.artist)
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
        comp = sentiment['compound']
    elif artist!='':
        output.append('Artist: '+music_result.name)
        output.append('----------------')
        output.append('Top 3 Songs for this artist:')
        i = 1
        pos = neg = neu = comp = 0
        for x in music_result.songs:
            output.append(str(i) + '. ' + x.title)
            sentiment = analyzer.polarity_scores(x.lyrics)
            pos += sentiment['pos']
            neg += sentiment['neg']
            neu += sentiment['neu']
            comp += sentiment['compound']
            i += 1
        pos = pos/3
        neg = neg/3
        neu = neu/3
        comp = comp/3
    elif song!='':
        output.append('Song: '+music_result.title+' by '+music_result.artist)
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
        comp = sentiment['compound']
    else:
        pos = 0
        neg = 0
        neu = 0
        comp = 0
    
    if quote != '':
        val = getquote(quote)
        pos += val[0]
        neg += val[1]
        neu += val[2]
        comp += val[3]

    #listify(movies)
    output.append('----------------')
    pos_p = str(round(pos*100,2))
    neg_p = str(round(neg*100,2))
    neu_p = str(round(neu*100,2))
    s1 = 'Your music choice is '+pos_p+'% positive, '+neg_p+'% negative, and '\
    +neu_p+'% neutral'
    s2 = 'Your music choice has a compound sentiment of '+str(round(comp,4))\
    +' ('+sent_type(comp)+')'
    output.append(s1)
    output.append(s2)
    output.append('----------------')
    output.append('Movie: ' + movie_result[0])
    output.append('Summary: ')
    output.append(movie_result[1])
    # output.append('----------------')
    # output.append('Your Movie Recommendations Are:')
    #dists = get_sent_dist(pos,neg)
    dists = get_sent_dist(comp)
    scores = get_scores(movie_result[1], dists)
    ten = print_ten(movie_result[0],scores)
    output = [output] + ten
    return output

def get_songs(artist):
    result = genius.search_artist(artist, max_songs=3)
    top3_lst = []
    i = 1
    for x in result.songs:
        top3_lst.append('    '+str(i) + '. ' + x.title)
        i += 1
    return top3_lst

def get_artist_sentiment(artist):
    result = genius.search_artist(artist, max_songs=3)
    pos = neg = neu = comp = 0
    i = 1
    for x in result.songs:
        sentiment = analyzer.polarity_scores(x.lyrics)
        pos += sentiment['pos']
        neg += sentiment['neg']
        neu += sentiment['neu']
        comp += sentiment['compound']
        i += 1
    pos = pos/3
    neg = neg/3
    neu = neu/3
    comp = comp/3
    return [pos, neg, neu, comp]


def find_music(artist= '', song=''):
    if song != '' and artist!='':
        result = genius.search_song(song, artist)
    elif artist != '':
        result = genius.search_artist(artist, max_songs=3)
    elif song!= '':
        result = genius.search_song(song)
    else:
        result = ''
    return result

def getquote(quote= ''):
    sentiment = analyzer.polarity_scores(quote)
    pos = sentiment['pos']
    neg = sentiment['neg']
    neu = sentiment['neu']
    comp = sentiment['compound']
    return [pos, neg, neu, comp]


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


# def listify(df):
#     genres = []
#     languages = []
#     countries = []
#     for x,y,z in zip(df['Genres'],df['Languages'],df['Countries']):
#         g = re.findall(': \"(.*?)\"', x)
#         l = re.findall(': \"(.*?)\"', y)
#         c = re.findall(': \"(.*?)\"', z)
#         genres.append(g)
#         languages.append(l)
#         countries.append(c)
#     df['Genres'] = genres
    #df['Languages'] = languages
    #df['Countries'] = countries


def get_scores(query,dists):
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
    scores = np.asarray([score/q_norm for score in scores])
    dists = np.asarray(dists)
    ratings = get_ratings()
    total_scores = (2*scores+.15*dists+.02*ratings)
    #total_scores = (2.5*scores+1*dists)
    result = sorted(tuple(zip(total_scores, docs)),reverse=True)
    return result[:10]


def sent_type(sent):
    res = ''
    if sent < -.95:
        res = 'extremely negative'
    elif sent < -.75:
        res = 'very negative'
    elif sent < -.3:
        res = 'negative'
    elif sent < -.05:
        res = 'slightly negative'
    elif sent < .05:
        res = 'neutral'
    elif sent < .3:
        res = 'slightly positive'
    elif sent < .75:
        res = 'positive'
    elif sent < .95:
        res = 'very positive'
    else:
        res = 'extremely positive'
    return res


'''def get_sent_dist(p1, n1):
    dists = []
    for p2,n2 in tuple(zip(movies['pos'], movies['neg'])):
        dists.append(math.sqrt((p2 - p1)**2 + (n2 - n1)**2))
    dists = max(dists)*np.ones(len(dists))-dists
    return dists'''


def get_sent_dist(comp):
    dists = []
    for c in movies['compound']:
        dists.append(abs(float(c)-comp))
    dists = max(dists)*np.ones(len(dists))-dists
    return dists


def get_ratings():
    ratings = np.asarray(movies['Rating'])
    ratings = ratings-5*np.ones(len(ratings))
    return ratings


def print_ten(movie,results):
    ten = []
    i = 1
    for (score,ind) in results:
        entry = []
        if movies['Title'][ind] != movie:
            entry.append(str(i)+'.')
            entry.append(movies['Title'][ind])
            entry.append(str(round(score, 4)))
            entry.append(movies['Summary'][ind])#[:400]+'...')
            entry.append(movies['Streaming Services'][ind])
            i+=1
        ten.append(entry)
    return ten


'''inv_idx = pd.read_csv('app/inv_idx.csv')
inv_idx.columns = ['word','docs','counts']
z = tuple(zip(inv_idx['word'],inv_idx['docs']))
idf = {a: math.log2(n_mov/(1+len(b))) for (a,b) in z if len(b)>=20 and len(b)/n_mov<=0.8}
word_to_index = {word:i for i, word in enumerate(inv_idx['word'])}
docs = [d.strip('[]').split(', ') for d in inv_idx['docs']]
inv_idx['docs'] = docs
counts = [c.strip('[]').split(', ') for c in inv_idx['counts']]
inv_idx['counts'] = counts'''
