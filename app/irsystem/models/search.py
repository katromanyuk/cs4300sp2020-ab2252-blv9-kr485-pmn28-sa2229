import numpy as np
import pandas as pd
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TOKEN = 'sD0C3epnJdfOQQK4eIC45dHl-Qv7DipToGpuj1n4WeuG5_LDP1HKn31w5Cn1lOux'
genius = lyricsgenius.Genius(TOKEN)
genius.verbose = False
genius.remove_section_headers = True

def find_music(query):
    output = []
    if ',' in query:
        query = query.split(',')
        result = genius.search_song(query[0], query[1])
        output.append('Song: '+result.title)
        output.append('----------------')
        output.append('Artist: '+result.artist)
        output.append('----------------')
        output.append('Lyrics: '+result.lyrics)
        output.append('----------------')
    else:
        result = genius.search_artist(query, max_songs=5)
        output.append('Artist: '+result.name)
        output.append('----------------')
        output.append('Top songs:')
        output.append('----------------')
        i = 1
        for song in result.songs:
            output.append(str(i)+'. '+song.title)
            i+=1
    return output

def find_movie(query):
    output = ['Movie: ...']
    return output
