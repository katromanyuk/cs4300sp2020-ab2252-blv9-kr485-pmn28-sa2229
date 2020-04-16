from . import *
from app.irsystem.models.search import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Clever Monkeys: Movie Recommendations Based on Music Preferences"
net_id = "Amber Baez: ab2252, Betsy Vasquez Valerio: blv9, " \
"Kateryna Romanyuk: kr485, Patrick Neafsey: pmn28, Shilpy Agarwal: sa2229"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search1')
	query2 = request.args.get('search2')
	if not query or not query2:
		output_message = ''
		data = ['Please fill out both fields']
	else:
		output_message = "Your search: " + query + " and " + query2
		music = find_music(query)
		movie = find_movie(query2)
		data= music + movie
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
