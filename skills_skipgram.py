from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.manifold import TSNE

# Word2Vec skipgram model creation
sentences = LineSentence('skill.tsv')
model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)

# T-SNE to reduce dimensionality to 2

# word_vectors = model.wv
# for i in word_vectors:
# 	print(i)
# print(word_vectors.Vocab)
# print(word_vectors, word_vectors.shape)

raw_vectors = np.genfromtxt('skill.tsv', delimiter="\t")[1:]
X_embedded = TSNE(n_components=2).fit_transform(raw_vectors)
np.savetxt('d3-scatterplot/embeddings.tsv', X_embedded, header='x\ty\tskill',delimiter="\t")

