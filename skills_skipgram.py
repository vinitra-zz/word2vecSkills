from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.manifold import TSNE

# Word2Vec skipgram model creation
sentences = LineSentence('skill.tsv')
model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)

# T-SNE to reduce dimensionality to 2

word_vectors = model.wv
skill_keys = word_vectors.vocab.keys()

embeddings = []
for key in skill_keys:
	embeddings.append(word_vectors[key])

# raw_vectors = np.genfromtxt('skill.tsv', delimiter="\t")[1:]
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
np.savetxt('d3-scatterplot/embeddings.tsv', X_embedded, header='x\ty\tskill\n',delimiter="\t")

