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


def find_movie(movie):
    output = ['Movie...']
    # find the movie plot summary using IMDB API
    #token = 'ce887dbd'
    #query = "http://www.omdbapi.com/?apikey=" + token + "&s=" + movie
    #params = {"r": "json"}
    #result = requests.request("GET", query)
    #plot = result[plot]
    # output.append(plot)

    return output
