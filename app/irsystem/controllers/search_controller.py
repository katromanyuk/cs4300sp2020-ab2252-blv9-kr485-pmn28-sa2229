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
    quote = request.args.get('search4')
    string = 'Your search: '
    output_message = ''
    if movie:
        if artist:
            string+=artist
        if song:
            string+= song
        if movie:
            string+=movie
        if quote:
            string+=quote
        output_message = string

        data = get_data(artist, song, movie)
    else:
        data = []
        output_messgae = "While all the other fields are optional, a movie is required. Please try again!"
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
