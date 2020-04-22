from . import *
from app.irsystem.models.search import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Clever Monkeys: Movie Recommendations Based on Music Preferences"
net_id = "Amber Baez: ab2252, Betsy Vasquez Valerio: blv9, " \
    "Kateryna Romanyuk: kr485, Patrick Neafsey: pmn28, Shilpy Agarwal: sa2229"


@irsystem.route('/', methods=['GET'])
def search():
    artist = request.args.get('search1')
    song = request.args.get('search2')
    movie = request.args.get('search3')
    if not artist or not movie:
        output_message = ''
        data = ['Please give us at least an artist and movie!']
    else:
        if not song:
            output_message = 'Your search: '+artist+', '+movie
        else:
            output_message = 'Your search: '+artist+', '+song+', '+movie
        data = get_data(artist, song, movie)
    return render_template('search2.html', name=project_name, netid=net_id, output_message=output_message, data=data)
