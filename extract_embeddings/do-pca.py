import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('file1', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--k', type=int, default=50, help='Number of components')
parser.add_argument('--story-name', type=str)
args = parser.parse_args()
print(args)

# Read df and embeddings
df = pd.read_csv(args.file1, sep=',', header=None)
# cols = [c for c in df.columns if c.isnumeric()]
# idxs = df.iloc[:, -1].notna()
# embeddings = df.loc[idxs, cols].values.astype(np.float)
if args.story_name=='monkey':
    embeddings = df.iloc[:, 5:-1].values.astype(np.float)
else:
    embeddings = df.iloc[:, 3:-1].values.astype(np.float)

assert embeddings.size > 0, 'Embeddings not loaded correctly'
assert embeddings.ndim == 2, 'Embeddings not loaded correctly'

pca = PCA(n_components=args.k, whiten=False, random_state=42)
reducedX = pca.fit_transform(embeddings)

# Rescale
reducedX = reducedX / np.sqrt(pca.explained_variance_)  # same as whiten
# reducedX = reducedX / pca.singular_values_

assert reducedX.shape[0] == embeddings.shape[0]

df2 = pd.read_csv(args.file1, sep=',', header=None, usecols=range(5))
# df2.drop(columns=cols, inplace=True)
# embeddings1 = np.full((df2.shape[0], args.k), np.nan)
# embeddings1[idxs] = reducedX

# Write out new datum with embeddings
df3 = pd.concat((df2, pd.DataFrame(reducedX)), axis=1)
df3.to_csv(args.output, index=False)
