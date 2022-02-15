import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from anndata import AnnData
from sklearn.decomposition import PCA


ba44_dir = '/mnt/Shared01/BA44/'
ba46_dir = '/mnt/Shared01/BA46/'
unified_dir = '/mnt/Shared01/BA44-46_unified/'

ba44_meta = pd.read_csv(os.path.join(ba44_dir, 'Splotch_Metadata.tsv'), 
	header=0, index_col=0, sep='\t')
ba46_meta = pd.read_csv(os.path.join(ba46_dir, 'BA46_samples.csv'),
	header=0, index_col=None, sep=',')

# Count files containing only genes shared between BA44 and BA46 samples.
ba44_countfiles = [os.path.join(unified_dir, 'raw_counts', Path(g).name + '.unified_cortex.tsv') for g in ba44_meta['Spaceranger output']]
ba46_countfiles = [os.path.join(unified_dir, 'raw_counts', g+'.unified_cortex.tsv') for g in ba46_meta['Spaceranger output']]



##### Perform pre-processing on all ST data: #####
# 1. Drop any spots with fewer than 100 UMIs.
# 2. Depth-normalize to 1e4 counts.
# 3. log(x+1)-transform count data.
# 4. Scale each gene based on mean/std across training (BA44) data.

def filtered_norm_logcounts(st_countfile, min_counts=100, target_sum=1e4):
	df = pd.read_csv(st_countfile, header=0, index_col=0, sep='\t')
	genes = pd.DataFrame(index=df.index)
	spots = pd.DataFrame(index=df.columns)
	adata = AnnData(X=df.values.T, obs=spots, var=genes)
	
	sc.pp.filter_cells(adata, min_counts=min_counts)    # Remove spots with <100 UMIs
	sc.pp.normalize_total(adata, target_sum=target_sum)	# Depth normalization
	sc.pp.log1p(adata) 

	df_pp = pd.DataFrame(adata.X.T, index=adata.var.index, columns=adata.obs.index)
	return df_pp


# Calculate per-gene mean/std across BA44 data
ba44_allcounts = []
for cfile in ba44_countfiles:
	df = filtered_norm_logcounts(cfile)
	ba44_allcounts.append(df.values)
ba44_allcounts = np.hstack(ba44_allcounts)

ba44_mean = ba44_allcounts.mean(axis=1)
ba44_std = ba44_allcounts.std(axis=1)

print(ba44_allcounts.shape)
print(ba44_mean.shape, ba44_std.shape)

# Pre-process and scale each ST array
for cfile in ba44_countfiles + ba46_countfiles:
	df = filtered_norm_logcounts(cfile)
	X = (df.values - ba44_mean[:, None]) / ba44_std[:, None]
	X = np.minimum(X, 10)  # Clip values to 10

	df_scaled = pd.DataFrame(X, columns=df.columns, index=df.index)

	# Write to CSV to save time
	outfile = cfile.replace('raw_counts', 'logscaled_counts')
	df_scaled.to_csv(outfile, sep='\t')



##### Fit PCA model to BA44 count data #####

# Scaled log normalized counts
ba44_countfiles = [os.path.join(unified_dir, 'logscaled_counts', Path(g).name + '.unified_cortex.tsv') for g in ba44_meta['Spaceranger output']]

ba44_allspots = None
for cfile in ba44_countfiles:
	print(cfile)
	df0 = pd.read_csv(cfile, header=0, index_col=0, sep='\t', nrows=2)
	float_cols = [c for c in df0 if df0[c].dtype == "float64"]
	float32_cols = {c: np.float32 for c in float_cols}
	df = pd.read_csv(cfile, header=0, index_col=0, sep='\t', 
		dtype=float32_cols)
	
	if ba44_allspots is None:
		ba44_allspots = df
	else:
		ba44_allspots = pd.concat((ba44_allspots, df), axis=1)
X = ba44_allspots.values.T
print(X.shape)

print('Calculating principal components...')
pca = PCA()
pca.fit(X)
pickle.dump(pca, open(os.path.join(unified_dir, 'trained_pca.p'), 'wb'))

npcs = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.5)[0][0] + 1
print('{} PCs explain >50% of variance in scaled log normalized counts'.format(npcs))


##### Apply trained model to preprocessed ST data #####