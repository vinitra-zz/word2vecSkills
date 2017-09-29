from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Number 1
sentences = LineSentence('skill.tsv')
model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)
