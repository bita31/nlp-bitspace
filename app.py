from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os

app = Flask(__name__)
CORS(app)

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def preprocess(teks):
    teks = teks.lower()
    teks = stopword_remover.remove(teks)
    teks = stemmer.stem(teks)
    return teks

@app.route('/nilai', methods=['POST'])
def nilai_esai():
    data = request.get_json()
    jawaban_siswa = data.get('jawaban_siswa', '')
    jawaban_referensi = data.get('jawaban_referensi', '')

    if not jawaban_siswa or not jawaban_referensi:
        return jsonify({'skor': 0})

    siswa_bersih = preprocess(jawaban_siswa)
    referensi_bersih = preprocess(jawaban_referensi)

    teks = [siswa_bersih, referensi_bersih]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(teks)
    kemiripan = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    skor = round(float(kemiripan[0][0]) * 100, 2)

    return jsonify({'skor': skor})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)