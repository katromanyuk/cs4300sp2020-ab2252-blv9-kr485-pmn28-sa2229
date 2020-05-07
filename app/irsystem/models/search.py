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


def get_data(artist, song, movie, quote, amazon, disney, hbo, hulu, netflix):
    output = []
    movie_result = find_movie(movie)
    #if movie_result=='ERROR':
    #    return ['We did not find the movie you searched for. Did you spell it correctly?']
    just_mov = False
    music_result = find_music(artist, song)

    pos = neg = neu = comp = 0

    if song != '':
        artist_name = music_result.artist
        song_disp = 'Song: ' + music_result.title + ' by ' + artist_name
        top3 = get_songs(artist_name)
        top3_disp = "Top 3 Songs by "+artist_name+": "
        sentiment = analyzer.polarity_scores(music_result.lyrics)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
        comp = sentiment['compound']
    elif artist != '':
        try:
            artist_name = music_result.name
            song_disp = 'Artist: ' + artist_name
            top3 = get_songs(artist_name)
            top3_disp = "Top 3 Songs by "+artist_name+": "
            sent_list = get_artist_sentiment(artist_name)
            pos = sent_list[0]
            neg = sent_list[1]
            neu = sent_list[2]
            comp = sent_list[3]
        except:
            song_disp = "Could not find your Artist. Did you spell it correctly?"
            top3 = []
            top3_disp = "No Artist Found"
    else:
        song_disp = "No Song Found"
        top3_disp = "No Artist Found"
        top3 = []

    if quote != '':
        val = getquote(quote)
        pos += val[0]
        neg += val[1]
        neu += val[2]
        comp += val[3]
        if top3_disp != "No Artist Found":
            pos /= 2
            neg /= 2
            neu /= 2
            comp /=2
        quote_disp = quote
    else:
        quote_disp = "N/A"

    if top3_disp == "No Artist Found" and quote_disp == "N/A":
        just_mov = True
        mov_sent = analyzer.polarity_scores(movie_result[1])
        pos = mov_sent['pos']
        neg = mov_sent['neg']
        neu = mov_sent['neu']
        comp = mov_sent['compound']

    pos_p = str(round(pos*100,2))
    neg_p = str(round(neg*100,2))
    neu_p = str(round(neu*100,2))

    s1 = pos_p+'% Positive, '+neg_p+'% Negative, and '+neu_p+'% Neutral'
    s2 = ''+str(round(comp,4))+' ('+sent_type(comp)+')'

    stream_list = get_stream_list(amazon,disney,hbo,hulu,netflix)

    row1 = [    [song_disp],        ['Movie: '+movie_result[0]]     ]
    row2 = [    [top3_disp]+top3,   ['Summary: ', movie_result[1]]  ]
    row3 = [    ['Sentiment Breakdown:', s1], ['Quote: ', quote_disp]  ]
    row4 = [    ['Compound Sentiment:', s2], ['Selected Streaming Services: ', stream_list]]
    output = [row1, row2, row3, row4]

    dists = get_sent_dist(comp)
    stream = get_stream_scores(amazon,disney,hbo,hulu,netflix)
    scores = get_scores(movie_result[1],dists,stream,just_mov)
    ten = print_ten(movie_result[0],scores)
    output = [output] + ten
    return output


def get_stream_list(amazon, disney, hbo, hulu, netflix):
    streaming = []
    if amazon=='1':
        streaming.append('Amazon')
    if disney=='1':
        streaming.append('Disney')
    if hbo=='1':
        streaming.append('HBO')
    if hulu=='1':
        streaming.append('Hulu')
    if netflix=='1':
        streaming.append('Netflix')
    if len(streaming)>0:
        result = (', '.join(streaming))
    else:
        result = 'N/A'
    return result


def get_stream_scores(amazon, disney, hbo, hulu, netflix):
    streaming = []
    if amazon=='1':
        streaming.append(movies['Amazon'])
    if disney=='1':
        streaming.append(movies['Disney'])
    if hbo=='1':
        streaming.append(movies['HBO'])
    if hulu=='1':
        streaming.append(movies['Hulu'])
    if netflix=='1':
        streaming.append(movies['Netflix'])
    if len(streaming)>0:
        scores = np.amax(np.asarray(streaming), axis=0)
    else:
        scores = np.zeros(n_mov)
    return scores


def get_songs(artist):
    result = genius.search_artist(artist, max_songs=3)
    top3_lst = []
    i = 1
    for x in result.songs:
        song_str = '    \t'+str(i)+'. '+x.title
        top3_lst.append(song_str)
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
        res_type = json[3]
        poster = json[4]
    else:
        return "ERROR"
    return (title, plot, review_imdb, res_type, poster)


def cleanjson(result):
    title = result[result.find("Title")+8:result.find("Year")-3]
    plot = result[result.find("Plot")+7:result.find("Language")-3]
    poster = result[result.find("Poster")+9:result.find("Ratings")-3]
    res_type = result[result.find("Type")+7:result.find("Type")+12]
    try:
        review_imdb = float(result[result.find(
        "imdbRating")+13:result.find("imdbVotes")-3])
    except:
        review_imdb = "N/A"
    try:
        review_rotten = float(result[result.find(
        'Source":"Rotten Tomatoes","Value":')+35: result.find('},{"Source":"Metacritic"')-2])
    except:
        review_rotten = "N/A"
    return [title, plot, review_imdb, res_type, poster]


def response(result):
    text = result[result.find('"Response":')+12:]
    return text.find('True') > -1


def get_scores(query,dists,stream,just_mov):
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
    ratings = np.asarray(movies['Rating'])
    if just_mov:
        total_scores = (1.3*scores+.04*dists+.03*ratings+.08*stream)
    else:
        total_scores = (1.3*scores+.08*dists+.03*ratings+.08*stream)
    result = sorted(tuple(zip(total_scores, docs)),reverse=True)
    return result[:15]


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


def get_sent_dist(comp):
    dists = []
    for c in movies['compound']:
        dists.append(abs(float(c)-comp))
    dists = max(dists)*np.ones(len(dists))-dists
    return dists


def get_ratings():
    ratings = np.asarray(movies['Rating'])
    #ratings = ratings-5*np.ones(len(ratings))
    return ratings


def print_ten(movie,results):
    ten = []
    i = 1
    for (score,ind) in results:
        entry = []
        if movies['Title'][ind] != movie:
            title = movies['Title'][ind]
            movie_result = find_movie(title)
            if movie_result!='ERROR' and movie_result[3]=='movie':
                summ = movie_result[1]
                if summ=='N/A':
                    summ = movies['Summary'][ind]
                #summ = movies['Summary'][ind]
                poster = movie_result[4]
                rate = movie_result[2]
            else:
                summ = movies['Summary'][ind]
                poster = ''
                rate = movies['Rating'][ind]
            summ = summ.replace('\\','')
            if rate==0:
                rate='N/A'
            entry.append(str(i)+'.')
            entry.append(title)
            entry.append(poster)
            entry.append(str(round(score, 4)))
            entry.append(summ)
            entry.append(movies['Streaming Services'][ind])
            entry.append(rate)
            i+=1
        if len(entry)>0:
            ten.append(entry)
    return ten
