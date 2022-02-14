import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

ba44_dir = '/mnt/home/adaly/ceph/datasets/BA44/'
ba46_dir = '/mnt/home/adaly/ceph/datasets/BA46/'

# Count files containing only genes shared between BA44 and BA46 samples.
ba44_countfiles = glob.glob(os.path.join(ba44_dir, 'spaceranger_output', '*', '*.unified_cortex.tsv'))
ba46_countfiles = glob.glob(os.path.join(ba46_dir, 'spaceranger_output', '*', '*.unified_cortex.tsv'))

# Perform pre-processing on all ST data:
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
for cfile in ba44_countfiles[:2]:
	df = filtered_norm_logcounts(cfile)
	ba44_allcounts.append(df.values)
ba44_allcounts = np.hstack(ba44_allcounts)

ba44_mean = ba44_allcounts.mean(axis=1)
ba44_std = ba44_allcounts.std(axis=1)

# Pre-process and scale each ST array
for cfile in ba44_countfiles + ba46_countfiles:
	df = filtered_norm_logcounts(cfile)
	X = (df.values - ba44_mean[:, None]) / ba44_std[:, None]
	X = np.minimum(X, 10)  # Clip values to 10

	print(cfile)
	print(X.min(), X.max())
	print(X.mean(axis=1))
	print(X.std(axis=1))

	df_scaled = pd.DataFrame(X, columns=df.columns, index=df.index)