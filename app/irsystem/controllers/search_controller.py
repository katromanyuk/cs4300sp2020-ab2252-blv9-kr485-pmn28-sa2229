from . import *
from app.irsystem.models.search import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Melodic Monkeys: Movie Recommendations Based on Music Preferences"
net_id = "Amber Baez: ab2252, Betsy Vasquez Valerio: blv9, " \
    "Kateryna Romanyuk: kr485, Patrick Neafsey: pmn28, Shilpy Agarwal: sa2229"

@irsystem.route('/about.html')
def go_to_about():
    return render_template('about.html')

@irsystem.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@irsystem.route('/', methods=['GET'])
def search():
    artist = request.args.get('artist')
    song = request.args.get('song')
    movie = request.args.get('movie')
    quote = request.args.get('quote')
    amazon = request.args.get('amazon')
    disney = request.args.get('disney')
    hbo = request.args.get('hbo')
    hulu = request.args.get('hulu')
    netflix = request.args.get('netflix')
    output_message = 'Please enter a movie to get results!'
    if not movie:
        data = []
    else:
        data = get_data(artist, song, movie, quote, amazon, disney, hbo, hulu, netflix)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
