from flask import render_template
from config import app
from flask import request

ROWS_PER_PAGE = 5


@app.route('/', methods=['GET'])
def index():
    context = {}
    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)
