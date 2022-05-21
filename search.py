import json
import joblib
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def write_json(prediksi, judul):
    response_json = {
      'code': 200,
      'kategori': prediksi,
      'judul': judul
    }
    with open("result.json", "w") as f:
        json.dump(response_json, f)

    return json.dumps(response_json)

def predict_and_search(headline_input):
    model = joblib.load("SAGA_LR_elasticnet_03_batched.pkl") #load pipeline
    def text_cleaning(judul):
        judul = judul.lower()                                                # casefolding
        judul = re.sub('\(.*?\) | \[.*?\]', ' ', judul)                      # kata di dalam kurung
        judul = re.sub('[%s]' % re.escape(string.punctuation), ' ', judul)   # punctuation
        judul = re.sub('[‘’“”…]', ' ', judul)                                # tanda kutip

        return judul

    factory = StopWordRemoverFactory()
    new_stopwords = ['yg', 'dgn', 'dlm', 'nya'] #in case ada judul yang pakai ini
    stopwords = factory.get_stop_words() + new_stopwords
    stopwords = factory.create_stop_word_remover()

    def stopwords_removal(judul):
        judul = stopwords.remove(judul)

        return judul

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    title = headline_input #take data in index i
    title = text_cleaning(title) #clean+fold
    title = stopwords_removal(title) #stopword remove
    title = stemmer.stem(title) #stem
    title = [title] #wrap in numpy format
    prediction = model.predict(title) #pipeline run -> tokenize vectorize tfidf then classify
    print('title is ', title)
    print('prediction is ', prediction)
    pResult = prediction[0] #result in [] so take data out
    response = write_json(pResult, title[0])
    result = "finish!"
    return result, response
