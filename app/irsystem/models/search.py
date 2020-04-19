import numpy as np
import pandas as pd
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

TOKEN = 'sD0C3epnJdfOQQK4eIC45dHl-Qv7DipToGpuj1n4WeuG5_LDP1HKn31w5Cn1lOux'
genius = lyricsgenius.Genius(TOKEN)
genius.verbose = False
genius.remove_section_headers = True


def find_music(artist, song=''):
    output = []
    if song != '':
        result = genius.search_song(song, artist)
        output.append('Song: '+result.title)
        output.append('Artist: '+result.artist)
        output.append('----------------')
        output.append('Lyrics: '+result.lyrics)
        output.append('----------------')
    else:
        result = genius.search_artist(artist, max_songs=3)
        output.append('Artist: '+result.name)
        output.append('----------------')
        output.append('Top 3 Songs for this artist:')
        i = 1
        for x in result.songs:
            output.append(str(i) + '. ' + x.title)
            i += 1
        output.append('----------------')
    return output


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
    output = []
    # find the movie plot summary using IMDB API
    token = 'ce887dbd'
    query = "http://www.omdbapi.com/?apikey=" + token + "&t=" + movie
    #query1 = "http://www.omdbapi.com/?apikey=" + token + "&s=" + movie
    params = {"r": "json", "plot": "full"}
    result = requests.get(query, params)
    if response(result.text):
        json = cleanjson(result.text)
        plot = json[1]
        title = json[0]
        review_imdb = json[2]
        review_rotten = json[3]
        output.append('Plot of' + movie + ':' +str(plot))
    else:
        output.append(
            "We did not find the movie you searched for. Did you spell it correctly?")
    return output
