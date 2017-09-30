from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.manifold import TSNE
import json

# Word2Vec skipgram model creation
sentences = LineSentence('skill.tsv')
model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)

# T-SNE to reduce dimensionality to 2

word_vectors = model.wv
skill_keys = word_vectors.vocab.keys()

json_file = open('skill_dict.json')
json_data_str = json_file.read()
json_data = json.loads(json_data_str)

skill_id_dict = {v: k for k, v in json_data.items()}

skills = []
embeddings = []
for key in skill_keys:
	embeddings.append(word_vectors[key])
	skills.append(skill_id_dict[key])

# raw_vectors = np.genfromtxt('skill.tsv', delimiter="\t")[1:]
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
X = np.hstack((X_embedded, np.array(skills).reshape(-1, 1)))
# with open('d3-scatterplot/embeddings.tsv','w') as f:
np.savetxt('d3-scatterplot/embeddings.tsv', X, header='x\ty\tskill', delimiter='\t', fmt="%s", comments='')

