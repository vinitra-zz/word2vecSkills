from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.manifold import TSNE
from datascience import *
import pandas as pd

# Number 1
sentences = LineSentence('assistment_id.tsv')
assistment_model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=10, iter=30)

word_vectors = assistment_model.wv
assistment_keys = word_vectors.vocab.keys()

skills_assignments = pd.read_csv("skill_builder_data_corrected.csv")
skills_assignments = skills_assignments[['assistment_id', 'skill_name']].to_dict()['assistment_id']

skill_id_dict = {v: k for k, v in skills_assignments.items()}

skills = []
embeddings = []
for key in assistment_keys:
    embeddings.append(word_vectors[key])
    skills.append(skill_id_dict[int(key)])

# raw_vectors = np.genfromtxt('skill.tsv', delimiter="\t")[1:]
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
X = np.hstack((X_embedded, np.array(skills).reshape(-1, 1)))
# with open('d3-scatterplot/embeddings.tsv','w') as f:
np.savetxt('d3-scatterplot/assistments.tsv', X, header='x\ty\tskill', delimiter='\t', fmt="%s", comments='')
