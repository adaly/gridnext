import os
import torch
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


#######################################################
# TODO:
# 1. Implement gene filtering/count data pre-processing
#######################################################


# Generate torch_geometric.Data representation of Visium array(s)
def visium_to_graphdata(spaceranger_outputs, annot_files=None, array_lbls=None, spaceranger_version=2.0):
	'''
	Parameters:
	----------
	spaceranger_outputs: (iter of) str
		path(s) to Spaceranger output directories containing results of Visium experiment(s)
	annot_files: (iter of) str
		path(s) to Loupe annotation files containing spot-level annotations for (each) Visium array
	array_lbls: iter of obj
		graph-level labels for each Visium array -- supercedes node-level annotations in annot_files
	spaceranger_version: float
		version of Spaceranger used in generation of output files

	Returns:
	-------
	gdat: torch_geometric.Data
		graph data representation of Visium array(s)
	'''
	if isinstance(spaceranger_outputs, str):
		x, A, pos, y = read_visium_graph(spaceranger_outputs, annot_files, spaceranger_version)
	else:
		if annot_files is None:
			annot_files = [None] * len(spaceranger_outputs)
		else:
			assert len(annot_files) == len(spaceranger_outputs), 'number of annot_files must match spaceranger_outputs'

		x_list, A_list, pos_list, y_list = [],[],[],[]
		cum_num_nodes = 0
		for srd, afile in zip(spaceranger_outputs, annot_files):
			x, A, pos, y = read_visium_graph(srd, afile, spaceranger_version)

			x_list.append(torch.tensor(x, dtype=torch.float))
			A_list.append(torch.tensor(A + cum_num_nodes, dtype=torch.long))
			pos_list.append(torch.tensor(pos, dtype=torch.long))
			y_list.append(y)

			# offset each subsequent array to obtain unique node numbers
			cum_num_nodes += x.shape[0]

		x = torch.cat(x_list, dim=0)
		A = torch.cat(A_list, dim=1)
		pos = torch.cat(pos_list, dim=0)
		if y_list[0] is not None:
			y = np.concatenate(y_list)

	# Encode graph label information
	if array_lbls is not None:
		le = LabelEncoder()
		le.fit(array_lbls)
		y_enc = torch.tensor(le.transform(array_lbls), dtype=torch.long)
	# Encode node label information
	elif annot_files is not None and annot_files[0] is not None:
		le = LabelEncoder()
		le.fit(y)
		y_enc = torch.tensor(le.transform(y), dtype=torch.long)
	# Unlabeled data
	else:
		y_enc = None

	# Instantiate geometric Data object
	gdat = Data(x=x, edge_index=A, y=y_enc, pos=pos)

	return gdat


# Reads Visium array from Spaceranger output and returns count, spatial information
def read_visium_graph(spaceranger_output, annot_file=None, spaceranger_version=2.0):
	'''
	Parameters:
	----------
	spaceranger_outputs: (iter of) str
		path to Spaceranger output directories containing results of Visium experiment
	annot_files: (iter of) str
		path to Loupe annotation files containing spot-level annotations for Visium array
	spaceranger_version: float
		version of Spaceranger used in generation of output files

	Returns:
	-------
	x: ndarray
		(spots, genes) count matrix
	A: ndarray
		(2, edges) sparse representation of spot adjacency matrix
	arr_coords: ndarray
		(spots, 2) array of Visium array coordinates
	y: ndarray or None
		(spots,) array of str annotations if annot_file provided, or None
	'''

	# Read in count matrix of dimension (spots, genes) indexed by Visium barcodes
	matrix_dir = os.path.join(spaceranger_output, "outs", "filtered_feature_bc_matrix")
	mat = mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

	features_path = os.path.join(matrix_dir, "features.tsv.gz")
	feature_ids = pd.read_csv(features_path, delimiter='\t', header=None, index_col=0).index

	barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
	barcodes = pd.read_csv(barcodes_path, delimiter='\t', header=None, index_col=0).index

	df_counts = pd.DataFrame.sparse.from_spmatrix(mat.T, index=barcodes, columns=feature_ids)

	# Read in position file, mapping ST barcodes to array/pixel coordinates
	if spaceranger_version >= 2.0:
		pos_file = os.path.join(spaceranger_output, 'outs', 'spatial', 'tissue_positions.csv')
		df_pos = pd.read_csv(pos_file, index_col=0, header=0)
	else:
		pos_file = os.path.join(spaceranger_output, 'outs', 'spatial', 'tissue_positions_list.csv')
		df_pos = pd.read_csv(pos_file, index_col=0, header=None,
			names=["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
	
	# only consider spots under tissue
	df_pos = df_pos.loc[df_pos['in_tissue']==1]
	df_counts = df_counts.loc[df_pos.index]

	# Load spot-level annotations, if provided
	if annot_file is not None:
		df_annot = pd.read_csv(annot_file, sep=',', header=0, index_col=0)

		# discard unannotated spots
		barcodes_annot = df_annot.index
		df_counts = df_counts.loc[barcodes_annot]
		df_pos = df_pos.loc[barcodes_annot]

		y = df_annot.iloc[:,0].values
	else:
		y = None

	# Calculate adjacency matrix
	arr_coords = df_pos[['array_row','array_col']].values
	dmat = squareform(pdist(arr_coords))
	# in Visium pseudo-hex indexing, adjacent spots <2 units apart
	A = np.vstack(np.where(np.logical_and(dmat > 0, dmat <= 2)))

	return df_counts.values, A, arr_coords, y


if __name__ == '__main__':
	data_dir = '/Users/adaly/Desktop/Visium/20220810_BA46/'
	arr_name = ['V009-CGND-HRA-02756-A', 'V009-CGND-HRA-02756-B'] 

	spaceranger_dir = [os.path.join(data_dir, 'spaceranger_output', a) for a in arr_name]
	annot_file = [os.path.join(data_dir, 'annotation', a+'.csv') for a in arr_name]
	array_lbls = ['tissue1', 'tissue2']
	
	visium_to_graphdata(spaceranger_dir, annot_files=annot_file)
