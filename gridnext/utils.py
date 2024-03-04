import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from gridnext.plotting import plot_confusion_matrix


############### Prediction functions ###############

# Outputs flattened list of model predictions (both integer labels and softmax vectors) for all foreground spots
def all_fgd_predictions(dataloader, model, f_only=False):
	true_vals, pred_vals, pred_smax = [], [], []

	# GPU support
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	for x,y in dataloader:
		x = x.to(device)
		y = y.to(device)

		with torch.no_grad():
			if f_only:
				outputs = model.patch_predictions(x)
			else:		
				outputs = model(x)
			outputs = outputs.permute((0,2,3,1))
			outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
			labels = torch.reshape(y, (-1,))
			outputs = outputs[labels > 0]
			labels = labels[labels > 0] - 1 # Once background elimintated, re-scale between [0,N]

			outputs = outputs.cpu()
			labels = labels.cpu()
			y_fgd_true = labels.data.numpy()
			y_fgd_pred = torch.argmax(outputs, axis=1).data.numpy()
			y_fgd_smax = F.softmax(outputs, dim=1).data.numpy()

			true_vals.append(y_fgd_true)
			pred_vals.append(y_fgd_pred)
			pred_smax.append(y_fgd_smax)

	true_vals = np.concatenate(true_vals)
	pred_vals = np.concatenate(pred_vals)
	pred_smax = np.concatenate(pred_smax)

	return true_vals, pred_vals, pred_smax


############### Visium and Splotch Helper functions ###############

# Convert from pseudo-hex indexing scheme used by Visium to "odd-right" hexagonal indexing
# (odd-numbered rows are implicitly shifted one half-unit right; vertical axis implicitly scaled by sqrt(3)/2)
def pseudo_hex_to_oddr(col, row):
    if row % 2 == 0:
        x = col/2
    else:
        x = (col-1)/2
    y = row
    return int(x), int(y)

# Invert above transformation
def oddr_to_pseudo_hex(col, row):
    if row % 2 == 0:
        x = 2*col
    else:
        x = 2*col + 1
    y = row
    return int(x), int(y)

# Convert from Visium's pseudo-hex indexing to true Cartesian coordinates (neighbors unit distance apart)
def pseudo_to_true_hex(col, row):
    x = col / 2
    y = row * np.sqrt(3) / 2
    return x, y

# Read paired count and annotation files and populate input/label ndarrays for GridNet.
def read_annotated_starray(count_file, annot_file=None, select_genes=None, 
    h_st=78, w_st=64, Visium=True, position_file=None, cfile_delim='\t', afile_delim='\t'):
    '''
    Parameters:
    ----------
    count_file: path
        path to Splotch-formatted count file -- tab-delimited, (genes x spots).
    annot_file: path
        path to annotation file in either:
        - Loupe (Visium) format: barcode,annotation columns, or
        - Splotch format: (spot_coords x annotations) binary one-hot matrix.
    filter_genes: iterable of str
        list of gene names to include, or None to include all.
    h_st: int
        number of rows in ST array.
    w_st: int
        number of columns in ST array.
    Visium: bool
        whether ST data is from Visium platform, and thus spots are hexagonally packed.
    position_file: path
        for Visium data, path to position file mapping barcodes to array coordinates.
        If provided, annotation data will be interpreted as Visium format.
    cfile_delim: char
        count file delimiter
    afile_delim: char
        annotation file delimiter

    Returns:
    ----------
    counts_grid: (h_st, w_st, n_genes) ndarray
        float array containing counts for each gene at each spot (Visium data are odd-right indexed 
        to fit square array).
    annots_grid: (h_st, w_st)
        parallel array containing annotation index for each spot:
        - string array (BG='') when Loupe formatted annotations provided (Visium=True, tissue_position),
        - integer array (BG=0) when Splotch formatted annotations provided (tissue_position=None)
    '''
    cmat = pd.read_csv(count_file, header=0, index_col=0, sep=cfile_delim)
    if select_genes is not None:
        cmat = cmat.loc[select_genes, :]
    n_genes, _ = cmat.shape
    
    if annot_file is not None:
        if position_file is not None:
            # For Loupe-formatted annotations, return annotation names because each class may not be represented
            # (perform encoding after seeing full data)
            coord_strs, annot_strs = read_annotfile(annot_file, position_file=position_file, Visium=True)
            str_len = np.max([len(a) for a in annot_strs])
            annots_grid = np.empty((h_st, w_st), dtype='U%d' % str_len)
            adict = dict(zip(coord_strs, annot_strs))
        else:
            # For Splotch-formatted annotations, all classes included and can do encodings on the fly
            coord_strs, annot_lbls = read_annotfile(annot_file, Visium=False, afile_delim=afile_delim)
            annots_grid = np.zeros((h_st, w_st), dtype=int)
            adict = dict(zip(coord_strs, annot_lbls))
    
    counts_grid = np.zeros((h_st, w_st, n_genes), dtype=float)
    
    for cstr in cmat.columns:
        if Visium:
            x_vis, y_vis = map(int, cstr.split('_'))
            x, y = pseudo_hex_to_oddr(x_vis, y_vis)
        else:
            x_car, y_car = map(float, cstr.split('_'))
            x, y = int(np.rint(x_car)), int(np.rint(y_car))
        
        # Only include annotated spots
        if annot_file is not None:
            if cstr in adict.keys():
                counts_grid[y, x] = cmat[cstr].values
                if annots_grid.dtype == int:
                    annots_grid[y, x] = adict[cstr] + 1  # reserve 0 for BG
                else:
                    annots_grid[y, x] = adict[cstr]
        else:
            counts_grid[y, x] = cmat[cstr].values
            annots_grid[y, x] = 0
    
    return counts_grid, annots_grid

# Write a Loupe-formatted file containing annotations for a given tissue.
def to_loupe_annots(annot_tensor, position_file, output_file, annot_names=None, zero_bg=True):
    positions = pd.read_csv(position_file, index_col=0, header=None,
      names=["in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"])
    barcodes = []
    annotations = []

    annot_tensor = annot_tensor.squeeze()

    for i in range(len(positions)):
        ent = positions.iloc[i]
        if ent['in_tissue']:
            x, y = pseudo_hex_to_oddr(ent['array_col'], ent['array_row'])
            
            # If zero_bg, foreground annotations are between 1 and N_AAR, with 0 reserved for BG
            a = annot_tensor[y, x] - int(zero_bg)

            if a < 0:
                annotations.append('')
            elif annot_names is not None:
                annotations.append(annot_names[a])
            else:
                annotations.append(a)
            barcodes.append(positions.index[i])

    df = pd.DataFrame({'Barcode': barcodes, 'AARs': annotations})
    df.to_csv(output_file, sep=',', index=False)


# Convert AnnData of a single Visium array to paired input (features, h_st, w_st) and (h_st, w_st) label tensors.
def anndata_to_grids(adata, labels, h_st=78, w_st=64, use_pcs=False, vis_coords=True):
    if not use_pcs:
        counts_grid = torch.zeros((len(adata.var), h_st, w_st))
    else:
        counts_grid = torch.zeros((use_pcs, h_st, w_st))
    labels_grid = torch.zeros((h_st, w_st))

    if use_pcs:
        dat = adata.obsm['X_pca'][:,:use_pcs]
    elif sparse.issparse(adata.X):
        dat = adata.X.todense()
    else:
        dat = adata.X

    for i, (x,y) in enumerate(zip(adata.obs.x, adata.obs.y)):
        if vis_coords:
            x, y = pseudo_hex_to_oddr(x, y)
        labels_grid[y,x] = labels[i] + 1

        '''
        if use_pcs:
            counts_grid[:,y,x] = torch.tensor(adata.obsm['X_pca'][i,:use_pcs])
        elif sparse.issparse(adata.X):
            counts_grid[:,y,x] = torch.tensor(np.array(adata.X[i,:].todense()))
        else:
            counts_grid[:,y,x] = torch.tensor(adata.X[i,:])
        '''
        counts_grid[:,y,x] = torch.tensor(dat[i,:])
        
    return counts_grid.float(), labels_grid.long()

# Read in an annotation file and return paired lists of coordinate strings and annotations
def read_annotfile(afile, position_file=None, afile_delim=',', Visium=True):
    if Visium:
        adat = pd.read_csv(afile, header=0, index_col=0, sep=afile_delim)
        pdat = visium_get_positions_fromfile(position_file)

        # Filter unannotated spots
        adat = adat[adat.iloc[:,0] != '']

        adat = adat.join(pdat, how='left')
        adat = adat.dropna()
        coord_strs = ['%d_%d' % (x,y) for x,y in zip(adat['array_col'], adat['array_row'])]
        annot_strs = adat.iloc[:,0].values

        return coord_strs, annot_strs
    
    else:
        adat = pd.read_csv(afile, header=0, index_col=0, sep=afile_delim)

        # Filter improperly annotated spots:
        adat = adat[adat.sum(axis=1) == 1]

        coord_strs = adat.columns
        annot_lbls = np.argmax(adat.values, axis=0)

        return coord_strs, annot_lbls

# Given Spaceranger directory, locate and read mapping of spot barcodes to array/pixel coordinates
def visium_get_positions(spaceranger_dir):
    position_path = visium_find_position_file(spaceranger_dir)
    positions = visium_get_positions_fromfile(position_path)
    return positions

# Given position file, read mapping of spot barcodes to array/pixel coordinates
def visium_get_positions_fromfile(position_file):
    # Infer Spaceranger version from structure of file
    spaceranger_version = 1 
    with open(position_file, 'r') as fh:
        line = next(iter(fh))
        if line.startswith('barcode'):
            spaceranger_version = 2

    if spaceranger_version >= 2:
        positions = pd.read_csv(position_file, index_col=0, header=0)
    else:
        positions = pd.read_csv(position_file, index_col=0, header=None,
            names=["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
    return positions

# Given Spaceranger directory, locate file mapping spot barcodes to array/pixel coordinates
def visium_find_position_file(spaceranger_dir):
    position_paths = [
        os.path.join("outs", "spatial", "tissue_positions.csv"),      # Spaceranger >=2.0
        os.path.join("outs", "spatial", "tissue_positions_list.csv")  # Spaceranger <2.0
    ]
    for pos_path in position_paths:
        if os.path.exists(os.path.join(spaceranger_dir, pos_path)):
            return os.path.join(spaceranger_dir, pos_path)
    raise ValueError("Cannot location position file for %s" % spaceranger_dir)

