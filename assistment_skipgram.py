from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Number 1
sentences = LineSentence('assistment_id.tsv')
assistment_model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)

word_vectors = assistment_model.wv
assistment_keys = word_vectors.vocab.keys()

embeddings = []
for key in assistment_keys:
	embeddings.append(word_vectors[key])
	