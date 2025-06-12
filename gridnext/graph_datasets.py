import os
import torch
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder

from squidpy.gr import spatial_neighbors
from torch_geometric.data import Data, InMemoryDataset


#########################################################
#        Visium AnnData to torch_geometric Data			#
#########################################################

# Create a single (graph) Data object out of a Visium AnnData
# - If the Visium AnnData contains multiple samples/batches, will treat as a single graph w/unconnected subgraphs
def visium_anndata_to_graphdata(adata, spatial_key='spatial', annot_col='annotation', annot_encoding=None, 
                                x_layer=None, x_obsm=None, **kwargs):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm containing node coordinates
    annot_col: str
        column in adata.obs containing spot/node-level annotations
    annot_encoding: list of str or None
        enumeration of all possible annotation classes to be used in integer coding.
        when None, defaults to ordered list of all annotations seen in adata.obs
    x_layer: str
        entry in adata.layers to use for node features (default: adata.X)
    x_obsm: str
        entry in adata.obsm to use for node features -- supercedes x_layer
    kwargs: additional arguments for compute_adjacency_radius

    Returns:
    -------
    gdat: torch_geometric.data.Data
        graph Data representation of adata
    '''
    # Access nodel-level features
    if x_layer is not None:
        X = adata.layers[x_layer]
    elif x_obsm is not None:
        X = adata.obsm[x_obsm]
    else:
        X = adata.X
    X = torch.tensor(X, dtype=torch.float)
    
    # Encode node-level labels
    if annot_encoding is not None:
        y = [list(annot_encoding).index(a) for a in adata.obs[annot_col].values]
    else:
        le = LabelEncoder().fit(adata.obs[annot_col])
        annot_encoding = le.classes_
        y = le.transform(adata.obs[annot_col])
    y = torch.tensor(y, dtype=torch.long)

    # Compute adjacencies and edge weights
    A, w = compute_adjacency_radius(adata, spatial_key, **kwargs)
    if A is None or w is None:
        return None
    A = torch.tensor(A, dtype=torch.long)
    w = torch.tensor(w, dtype=torch.float).T

    # Extract spot coordinates
    c = torch.tensor(adata.obsm[spatial_key], dtype=torch.float)
    
    gdat = Data(x=X, edge_index=A, edge_attr=w, y=y, pos=c)
    return gdat

# Create an iterable of (graph) Data objects, one for each sample (batch) in a Visium AnnData
# - If the Visium AnnData contains multiple samples/batches, treats each as a separate graph (to be passed to DataLoader)
def visium_anndata_to_graphdataset(adata, spatial_key='spatial', batch_col='array', annot_col='annotation', annot_encoding=None, 
                                   x_layer=None, x_obsm=None, **kwargs):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm containing node coordinates
    batch_col: str
        column in adata.obs denoting individual samples/batches, each of which will comprise a separate graph
    annot_col: str
        column in adata.obs containing spot/node-level annotations
    annot_encoding: list of str or None
        enumeration of all possible annotation classes to be used in integer coding.
        when None, defaults to ordered list of all annotations seen in adata.obs
    x_layer: str
        entry in adata.layers to use for node features (default: adata.X)
    x_obsm: str
        entry in adata.obsm to use for node features -- supercedes x_layer
    kwargs: additional arguments for compute_adjacency_radius

    Returns:
    -------
    gdat_list: list of torch_geometric.data.Data
        list of Data representing each individual sample/batch in the AnnData
    '''
    # Create a unified encoding for annotations (if not provided)
    if annot_encoding is None:
        le = LabelEncoder().fit(adata.obs[annot_col])
        annot_encoding = le.classes_

    # Create separate graph data object for each independent batch
    gdat_list, batch_names = [], []
    for b in adata.obs[batch_col].unique():
        adata_sub = adata[adata.obs[batch_col]==b]
        if len(adata_sub) > 0:
            gdat = visium_anndata_to_graphdata(adata_sub, spatial_key, annot_col, annot_encoding, x_layer, x_obsm, **kwargs)
            if gdat is None or not gdat.validate():
                print('Skipping batch %s (%d nodes) -- invalid graph' % (b, len(adata_sub)))
            else:
                gdat_list.append(gdat)
                batch_names.append(b)
            
    return gdat_list, batch_names

'''
Because graphs can have variables numbers of nodes/edges, we cannot use basic PyTorch Datasets to do minibatching.

Instead, Geometric provides two abstract base classes:
1. InMemoryDataset -- for datasets small enough to fit in CPU memory
2. Dataset -- for datasets that should be partially read into CPU memory

Using a subset of these classes, a DataLoader can create mini-batches by:
- Block-diagonal stacking the adjacency matrices (easy in COO sparse format), and
- Concatenating node/target features in the node dimension
https://pytorch-geometric.readthedocs.io/en/2.6.1/advanced/batching.html
'''
class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.', transform=None, pre_transform=None)
        self.data, self.slices = self.collate(data_list)

    def get(self, idx):
        return super().get(idx)

#########################################################
#        Squidy-based adjacency calculations			#
#########################################################

# Copies node coordinates to a desired key in adata.obsm (required for Squidpy's spatial_neighbors)
def add_spatial(adata, spatial_key, x_col, y_col, pseudo_hex=False):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm in which to store node coordinates
    x_col, y_col: str, str
        columns in adata.obs containing x and y coordinates (respectively) of nodes; only needed if spatial_key not set
    pseudo_hex: bool
        whether coordinates follow Visium's pseudo-hex scheme (default False: cartesian coordinates)
        if True, coordinates will be transformed to Cartesian for storage in adata.obsm
    '''
    # Store node locations in adata.obsm
    for c in [x_col, y_col]:
        if c not in adata.obs:
            raise ValueError('No entry for %s in obs metadata' % c)
    true_coords = adata.obs[[x_col, y_col]].values.copy().astype(float)
    if pseudo_hex:
        true_coords[:,0] = true_coords[:,0] * 0.5          # halve horizontal dimension
        true_coords[:,1] = true_coords[:,1] * np.sqrt(3)/2 # scale vertical dimension to achieve unit adjacencies
    adata.obsm[spatial_key] = true_coords

# Compute adjacencies for an AnnData object potentially spanning multiple samples (batches)
def compute_adjacency_radius(adata, spatial_key='spatial', x_col=None, y_col=None, batch_col=None, 
                             radius=1.01, unit_dist_um=100, pseudo_hex=False):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm in which to find/store node coordinates
    x_col, y_col: str, str
        columns in adata.obs containing x and y coordinates (respectively) of nodes (if spatial_key not set)
    batch_col: str
        column in adata.obs denoting independent samples (e.g., Visium arrays)
    radius: float
        radius in coordinate system within which nodes are considered neighbors 
        (default 1.01 to account for rounding errors in pseudo-visium conversion)
    unit_dist_um: float
        physical distance (in um) corresponding to unit distance in coordinate system
    pseudo_hex: bool
        whether coordinates follow Visium's pseudo-hex scheme (default False: cartesian coordinates)
        if True, coordinates will be transformed to Cartesian for storage in adata.obsm

    Returns:
    -------
    A: [2, num_edges] ndarray
        computed edges in COO format
    w: [num_edges,] ndarray
        physical distances (um) corresponding to each edge
    '''
    if spatial_key not in adata.obsm:
        add_spatial(adata, spatial_key, x_col, y_col, pseudo_hex)
    else:
        assert adata.obsm[spatial_key].shape == (len(adata), 2)
    C, D = spatial_neighbors(adata, spatial_key=spatial_key, library_key=batch_col, coord_type='generic', radius=radius, copy=True)
    max_k = C.sum(axis=0).max()
    if max_k == 0:
        return None, None
    
    A = C.nonzero()
    w = D[A] * unit_dist_um
    return np.array(A), w

#########################################################
# DEFUNCT												#
#########################################################

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
		bc_shared = df_annot.index.intersection(df_pos.index)
		df_counts = df_counts.loc[bc_shared]
		df_pos = df_pos.loc[bc_shared]

		y = df_annot.iloc[:,0].values
	else:
		y = None

	arr_coords = df_pos[['array_col','array_row']].values
	
	# transform coordinates from "pseudo-hex":
	true_coords = arr_coords.copy().astype(float)
	true_coords[:,0] = true_coords[:,0] * 0.5          # halve horizontal dimension
	true_coords[:,1] = true_coords[:,1] * np.sqrt(3)/2 # scale vertical dimension to achieve unit adjacencies

	# Calculate adjacency matrix from distance matrix
	# TODO: cost can be reduced from N^2 to N by enumerating all possible neighbors for each point and intersecting
	#   with list of coordinates actually present in data. 
	dmat = squareform(pdist(true_coords))
	# after transforming from pseudo-hex indexing, adjacent spots 1 units apart
	A = np.vstack(np.where(np.logical_and(dmat > 0, dmat <= 1)))

	return df_counts.values, A, arr_coords, y


if __name__ == '__main__':
	data_dir = '/Users/adaly/Desktop/Visium/20220810_BA46/'
	arr_name = ['V009-CGND-HRA-02756-A', 'V009-CGND-HRA-02756-B'] 

	spaceranger_dir = [os.path.join(data_dir, 'spaceranger_output', a) for a in arr_name]
	annot_file = [os.path.join(data_dir, 'annotation', a+'.csv') for a in arr_name]
	array_lbls = ['tissue1', 'tissue2']
	
	gdat = visium_to_graphdata(spaceranger_dir, annot_files=annot_file)

	# Check that edges are symmetric and each node has no more than six neighbors
	neighbors = np.zeros(gdat.edge_index.max()+1)
	for (x,y) in gdat.edge_index.T:
		assert x != y
		neighbors[x] += 0.5
		neighbors[y] += 0.5

	assert np.max(neighbors) <= 6

