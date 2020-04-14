from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import lyricsgenius

TOKEN = 'sD0C3epnJdfOQQK4eIC45dHl-Qv7DipToGpuj1n4WeuG5_LDP1HKn31w5Cn1lOux'
genius = lyricsgenius.Genius(TOKEN)
genius.verbose = False
genius.remove_section_headers = True

project_name = "Clever Monkeys: Movie Recommendations Based on Music Preferences"
net_id = "Amber Baez: ab2252, Betsy Vasquez Valerio: blv9, " \
"Kateryna Romanyuk: kr485, Patrick Neafsey: pmn28, Shilpy Agarwal: sa2229"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		query = query.split(',')
		result = genius.search_song(query[0], query[1])
		song = result.title
		artist = result.artist
		output_message = "Song: " + song + "\nArtist: " + artist
		data = result.lyrics
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
