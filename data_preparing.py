import numpy as np
import nltk
import pymorphy2
import json
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
#Расскомментить перед первым использованием
#nltk.download('punkt')
#nltk.download('stopwords')
from neural_network import *

def encode_sentence(text, vocab2index, N=200):
    #tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return [encoded, length]


def data_prepare(language_dict, ru=False, en=False, es=False):
    if ru:
        stop_words = set(stopwords.words('russian'))
        lemmatizer = pymorphy2.MorphAnalyzer()
    elif en:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    elif es:
        stop_words = set(stopwords.words('spanish'))
        lemmatizer = nltk.stem.SnowballStemmer('spanish')

    dict_prepared = []

    for text in tqdm(language_dict):
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'[0-9]', '', text)

        word_tokens = word_tokenize(text)
        word_tokens = [w for w in word_tokens if not w in stop_words]
        if ru:
            word_tokens = [lemmatizer.parse(w)[0].normal_form for w in word_tokens]
        elif en:
            word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
        elif es:
            word_tokens = [lemmatizer.stem(w) for w in word_tokens]

        word_tokens = [w for w in word_tokens if not w in stop_words]

        filtered_text = ' '.join(word_tokens)
        dict_prepared.append(filtered_text)
    return dict_prepared

def get_rating(all_text) :

    with open("vocab2index.json", encoding='utf-8') as fh:
        vocab2index = json.load(fh)
    ru_prepared = data_prepare(all_text, ru = True)
    ans = []
    for text in ru_prepared :
        ans.append(encode_sentence(text.split(), vocab2index))
    vocab_size = 24582
    return get_seq_rating(ans, vocab_size) + 1

if __name__ == '__main__' :
    text = ['отличный телефон, понравился и мне и все друзьям, прочный, долго держит зарядку и имеет хорошую камеру, в общем остался доволен',
            'Достоинства: Экран \n\nНедостатки: Плохое качество корпуса,через пол года окантовка начела осыпатся,и отремонтировать не реально,нет в продаже',
            'полная херня, никогда не покупайте здесь. Очень тормозит, постоянно приходится перезагружать, в общем отстой'
            ]
    print(get_rating(text))

