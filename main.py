from flask import Flask, request, render_template
from Universal_Test import next_word
app = Flask(__name__)

def hello_world():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def hello_world_post():
    text = request.form['text']
    return next_word(text)

if __name__ == '__main__':
    app.run('localhost', port=8080, debug=True)

