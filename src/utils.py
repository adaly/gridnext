import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from plotting import plot_confusion_matrix


# For each batch in dataloader, calculates model prediction.
# Returns a flattened, foreground-masked (true label > 0) list of spot predictions.
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
    if col % 2 == 0:
        x = col/2
    else:
        x = (col-1)/2
    y = row
    return int(x), int(y)

# Read paired count and annotation files and populate input/label ndarrays for GridNet.
def read_annotated_starray(count_file, annot_file=None, select_genes=None, 
    h_st=78, w_st=64, Visium=True, cfile_delim='\t', afile_delim='\t'):
    '''
    Parameters:
    ----------
    count_file: path
        path to Splotch-formatted count file -- tab-delimited, (genes x spots).
    annot_file: path
        path to Splotch-formatted annotation file -- tab-delimited, one-hot encoding (annotations x spots).
    filter_genes: iterable of str
        list of gene names to include, or None to include all.
    h_st: int
        number of rows in ST array.
    w_st: int
        number of columns in ST array.
    Visium: bool
        whether ST data is from Visium platform, and thus spots are hexagonally packed.

    Returns:
    ----------
    counts_grid: (h_st, w_st, n_genes) ndarray
        odd-right indexed float array containing counts for each gene at each Visium spot.
    annots_grid: (h_st, w_st)
        odd-right indexed integer array containing annotation index for each Visium spot (0=BG).
    gene_names: (n_genes,) ndarray
        names of genes in counts_grid.
    annot_names: (n_annots,) ndarray
        names of foreground annotations -- mapping of annots_grid[annots_grid > 0]-1.
    '''
    cmat = pd.read_csv(count_file, header=0, index_col=0, sep=cfile_delim)
    if select_genes is not None:
        cmat = cmat.loc[select_genes, :]
    n_genes, _ = cmat.shape
    
    if annot_file is not None:
        amat = pd.read_csv(annot_file, header=0, index_col=0, sep=afile_delim)
        annot_names = amat.index.values
    else:
        amat = None
        annot_names = []
    
    counts_grid = np.zeros((h_st, w_st, n_genes), dtype=float)
    annots_grid = np.zeros((h_st, w_st), dtype=int)
    
    for cstr in cmat.columns:
        if Visium:
            x_vis, y_vis = map(int, cstr.split('_'))
            x, y = pseudo_hex_to_oddr(x_vis, y_vis)
        else:
            x_car, y_car = map(float, cstr.split('_'))
            x, y = int(np.rint(x_car)), int(np.rint(y_car))
        
        # Only include annotated spots
        if amat is not None:
            if cstr in amat.columns and np.sum(amat[cstr].values) > 0:
                counts_grid[y, x] = cmat[cstr].values
                annots_grid[y, x] = np.argmax(amat[cstr].values) + 1
        else:
            counts_grid[y, x] = cmat[cstr].values
            annots_grid[y, x] = 0
    
    return counts_grid, annots_grid, cmat.index.values, annot_names

# Write a Loupe-formatted file containing annotations for a given tissue.
def to_loupe_annots(annot_tensor, position_file, output_file, annot_names=None):
    positions = pd.read_csv(position_file, index_col=0, header=None,
      names=["in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"])
    barcodes = []
    annotations = []

    annot_tensor = annot_tensor.squeeze()

    for i in range(len(positions)):
        ent = positions.iloc[i]
        if ent['in_tissue']:
            x, y = pseudo_hex_to_oddr(ent['array_col'], ent['array_row'])
            
            # foreground annotations are between 1 and N_AAR, with 0 reserved for BG
            a = annot_tensor[y, x] - 1

            if a < 0:
                annotations.append('')
            elif annot_names is not None:
                annotations.append(annot_names[a])
            else:
                annotations.append(a)
            barcodes.append(positions.index[i])

    df = pd.DataFrame({'Barcode': barcodes, 'AARs': annotations})
    df.to_csv(output_file, sep=',', index=False)
