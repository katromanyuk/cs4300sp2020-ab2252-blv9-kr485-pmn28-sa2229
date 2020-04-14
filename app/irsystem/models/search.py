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
    if query.index(',')==-1:
        result = genius.search_artist(query, max_songs=5)
        output.append('Artist: '+result.name)
        output.append('Top songs:')
        for song in result.songs:
            output.append(song.title)
    else:
        query = query.split(',')
        result = genius.search_song(query[0], query[1])
        output.append('Song: '+result.title)
        output.append('Artist: '+result.artist)
    return output
