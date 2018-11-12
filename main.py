import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = set(stopwords.words('english'))


def get_words(txt):
    return filter(
        lambda x: x not in STOP_WORDS,
        re.findall(r'\b(\w+)\b', txt)
    )


def parse_sentence_words(input_file_names):
    """Returns a list of a list of words. Each sublist is a sentence."""
    sentence_words = []
    for file_name in input_file_names:
        for line in open(file_name):
            line = line.strip().lower()
            line = line.decode('unicode_escape').encode('ascii', 'ignore')
            sent_words = map(get_words, sent_tokenize(line))
            sent_words = filter(lambda sw: len(sw) > 1, sent_words)
            if len(sent_words) > 1:
                sentence_words += sent_words
    return sentence_words


# You would see five .txt files after unzip 'text.zip'
input_file_names = ["text/001ssb.txt", "text/002ssb.txt", "text/003ssb.txt",
                    "text/004ssb.txt", "text/005ssb.txt"]
GOT_SENTENCE_WORDS = parse_sentence_words(input_file_names)

# size: the dimensionality of the embedding vectors.
# window: the maximum distance between the current and predicted word within a sentence.
model = Word2Vec(GOT_SENTENCE_WORDS, size=128, window=3, min_count=5, workers=4)
model.wv.save_word2vec_format("got_word2vec.txt", binary=False)

i = model.wv.index2word.index("stark")
d = model.wv.index2word.index("king")
print model.wv.vectors[i] - model.wv.vectors[d]
