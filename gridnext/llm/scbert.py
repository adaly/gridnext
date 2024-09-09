import os
import pkgutil
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from io import StringIO
from scipy import sparse
from gridnext.llm import PerformerLM

import torch
from torch import nn


# Preprocess raw count data for input to scBERT model
def preprocess_scbert(adata, target_depth=1e4, counts_layer=None, min_genes=None, min_depth=None, 
	gene_symbols=None, target_genes=None):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing raw count data
	target_depth: float
		number of counts to normalize each spot to
	counts_layer: str or None
		layer of adata containing raw counts, or "None" to default to adata.X
	min_genes: int or None
		filter spots with fewer than min_genes
	min_depth: int or None
		filter spots with fewer than min_counts (prior to depth normalization)
	gene_symbols: str or None
		column name in adata.var storing gene_symbols matching target_genes
	target_genes: path or None
		path to single-column CSV file containing ordered list of gene names to pull from adata,
		or "None" to default to the default list of gene2vec.
	'''
	if target_genes is None:
		ref_data = pkgutil.get_data('gridnext.llm', 'gene2vec_names.csv').decode('utf-8')
		ref_data = StringIO(ref_data)
	else:
		ref_data = target_genes
	ref_names = pd.read_csv(ref_data, header=None, index_col=0).index

	if counts_layer is None:
		X = adata.X
	else:
		X = adata.layers[counts_layer]
	counts = sparse.lil_matrix((X.shape[0],len(ref_names)),dtype=np.float32)
	ref = ref_names.tolist()

	if gene_symbols is not None:
		obj = adata.var[gene_symbols].tolist()
	else:
		obj = adata.var_names.tolist()

	# ensure overlap between adata.var and target_genes
	if len(set(obj) & set(ref)) == 0:
		raise ValueError("No matches to target_genes in reference -- check indexing of adata.var and/or set gene_symbols argument")

	for i in range(len(ref)):
		if ref[i] in obj:
			loc = obj.index(ref[i])
			counts[:,i] = X[:,loc]

	counts = counts.tocsr()
	new = ad.AnnData(X=counts)
	new.var_names = ref
	new.obs_names = adata.obs_names
	new.obs = adata.obs

	if min_genes is not None or min_depth is not None:
		sc.pp.filter_cells(new, min_genes=min_genes, min_counts=min_depth)

	sc.pp.normalize_total(new, target_sum=target_depth)
	sc.pp.log1p(new, base=2)

	return new
	

# scBERT model class; functional wrapper around PerformerLM
class scBERT(PerformerLM):
	'''
	Parameters:
	----------
	n_genes: int
		number of genes
	bin_num: int
		number of bins to discretize log gene expression into
	dim: int
		dimension of tokens in Performer model embedding
	depth: int
		number of layers in Performer model
	heads: int
		number of attention heads in each Performer layer
	local_attn_heads:
		number of heads devoted to local attention (remaning are global)
	gv2_pos_embed: bool
		whether to use gene2vec positional embedding
	n_classes: int or None
		number of classes in final classification layer, or "None" for a model that will be self-trained
	
	'''
	def __init__(self, n_genes=16906, bin_num=5, dim=200, depth=6, heads=10, local_attn_heads=0, 
		g2v_pos_embed=True, n_classes=None):
		super(scBERT, self).__init__(num_tokens=bin_num+2, dim=dim, depth=depth, max_seq_len=n_genes+1,
			heads=heads, local_attn_heads=local_attn_heads, g2v_position_emb=g2v_pos_embed)

		if n_classes is not None:
			self.to_out = AttentionClassifier(dropout=0, h_dim=128, out_dim=n_classes)
			self.finetune()

		self.bin_num = bin_num
		self.n_classes = n_classes

	# Prior to passing log-normalized gene expression data to the network:
	# 1. Discretize into bin_num bins
	# 2. Add an additional feature to the end
	def forward(self, x):
		x[x > self.bin_num] = self.bin_num
		x = x.long()
		new_feat = torch.zeros((x.shape[0],1), dtype=torch.long)
		x = torch.cat((x, new_feat), dim=-1)
		return super(scBERT, self).forward(x)

	# Turn off gradient for Performer parameters for fine-tuning of classification layer only
	def finetune(self):
		for param in self.parameters():
			param.requires_grad = False
		for param in self.norm.parameters():
			param.requires_grad = True
		for param in self.performer.net.layers[-2].parameters():
			param.requires_grad = True
		for param in self.to_out.parameters():
			param.requires_grad = True 
		

# Final layer of Performer that outputs classification logits
class AttentionClassifier(nn.Module):
	def __init__(self, in_features = 16906+1, dropout = 0., h_dim = 100, out_dim = 10):
		super(AttentionClassifier, self).__init__()
		self.conv1 = nn.Conv2d(1, 1, (1, 200))
		self.act = nn.ReLU()
		self.fc1 = nn.Linear(in_features=in_features, out_features=512, bias=True)
		self.act1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
		self.act2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)
		self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

	def forward(self, x):
		x = x[:,None,:,:]
		x = self.conv1(x)
		x = self.act(x)
		x = x.view(x.shape[0],-1)
		x = self.fc1(x)
		x = self.act1(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		x = self.act2(x)
		x = self.dropout2(x)
		x = self.fc3(x)
		return x
