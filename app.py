"""
Search Engine Web
Author: Zhao Yuqi
Date: 2022.6.17
"""

from flask import Flask, render_template
from searcher import JpnSearchEngine
import time


app = Flask(__name__)
search = JpnSearchEngine('dataset/leads.org.txt', length=3600)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search/<data>')
def query(data):
    now = time.perf_counter()

    top_n = 3
    res = search.get_top_n(data, n=top_n)

    top_n = 0 if res is None else min(top_n, len(res))
    success = False if res is None else True
    res_dic = {
        'success': success,
        'top_n': top_n,
        'query': data,
        'result': res
    }

    print('Time cost per query: ', time.perf_counter() - now, 's')
    return render_template('result.html', data=res_dic)


if __name__ == '__main__':
    app.run()
