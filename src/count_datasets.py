import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, TensorDataset


############### Read full datasets into memory ###############

def load_count_dataset(count_files, annot_files, select_genes=None):
    count_data_full = []
    annot_data_full = []

    for cf, af in zip(count_files, annot_files):
        adat = pd.read_csv(af, header=0, index_col=0, sep='\t')

        # Remove all unannotated or mis-annotated columns
        for cstr in adat.columns:
            if adat[cstr].values.sum() != 1:
                adat.drop(columns=cstr, inplace=True)

        cdat = pd.read_csv(cf, header=0, index_col=0, sep='\t')

        for cstr in adat.columns:
            if cstr in cdat.columns:
                if select_genes is not None:
                    count_data_full.append(cdat.loc[select_genes,cstr].values.squeeze())
                else:
                    count_data_full.append(cdat[cstr].values.squeeze())
                annot_data_full.append(adat[cstr].values.argmax())

    count_data_full = torch.tensor(count_data_full).float()
    annot_data_full = torch.tensor(annot_data_full).long()
    
    return TensorDataset(count_data_full, annot_data_full)

def load_count_grid_dataset(count_files, annot_files, select_genes=None, 
    h_st=78, w_st=64, Visium=True):
    count_data_full = []
    annot_data_full = []
    
    for cf, af in zip(count_files, annot_files):
        counts_grid, annots_grid, _, _ = read_annotated_starray(cf, af, select_genes=select_genes,
            h_st=h_st, w_st=w_st, Visium=Visium)
        
        count_data_full.append(counts_grid)
        annot_data_full.append(annots_grid)
    
    count_data_full = torch.tensor(count_data_full).permute(0,3,1,2)  # Reshape to channels-first
    annot_data_full = torch.tensor(annot_data_full)
    
    return TensorDataset(count_data_full.float(), annot_data_full.long())


############### Map-style PyTorch datasets ###############

class CountDataset(Dataset):
    # For independent classification of spots based on 1d expression vectors.

    def __init__(self, count_files, annot_files, select_genes=None,
        cfile_delim='\t', afile_delim='\t'):
        super(CountDataset, self).__init__()
                
        if not len(count_files) == len(annot_files):
            raise ValueError('Length of count_files and annot_files must match.')

        self.cfile_delim = '\t'
        self.afile_delim = '\t'
        
        self.countfile_mapping = []
        self.annotations = []
        self.cind_mapping = []
        
        # Construct unique integer index for all annotated patches
        for cf, af in zip(count_files, annot_files):
            with open(cf, 'r') as fh:
                counts_header = next(fh).strip('\n').split(self.cfile_delim)
                
                adat = pd.read_csv(af, header=0, index_col=0, sep=self.afile_delim)

                for cstr in adat.columns:
                    # Skip over unannotated or mis-annotated spots
                    if not cstr in counts_header:
                        print(af, cstr, 'missing')
                        continue
                    counts_ind = counts_header.index(cstr)

                    if not np.sum(adat[cstr]) == 1:
                        print(af, cstr, 'improper annotation')
                        continue

                    self.annotations.append(np.argmax(adat[cstr]))
                    self.countfile_mapping.append(cf)
                    self.cind_mapping.append(counts_ind)
        
        self.select_genes = select_genes
        
    def __len__(self):
        return len(self.cind_mapping)

    def __getitem__(self, idx):
        if self.select_genes is not None:
            return self._get_select(idx)
        
        df = pd.read_csv(self.countfile_mapping[idx], sep=self.cfile_delim, header=0, index_col=0,
                        usecols=[0, self.cind_mapping[idx]])
        count_vec = torch.from_numpy(df.values.squeeze())

        return count_vec.float(), torch.tensor(self.annotations[idx]).long()
    
    # More efficient implementation when we are interested in only a select set of genes.
    def _get_select(self, idx):
        count_vec = []
        
        with open(self.countfile_mapping[idx], 'r') as fh:
            header = next(fh)
            
            for line in fh:
                keep = False 
                for g in self.select_genes:
                    if line.startswith(g+self.cfile_delim):
                        keep = True
                        break
                if keep:
                    tokens = line.split(self.cfile_delim)
                    count_vec.append(float(tokens[self.cind_mapping[idx]]))
            
        return torch.tensor(count_vec).float(), torch.tensor(self.annotations[idx]).long()

class CountGridDataset(Dataset):
    # For registration of entire ST arrays based on 3d expression maps.

    def __init__(self, count_files, annot_files, select_genes=None, 
        h_st=78, w_st=64, Visium=True, cfile_delim='\t', afile_delim='\t'):
        super(CountGridDataset, self).__init__()
        
        if not len(count_files) == len(annot_files):
            raise ValueError('Length of count_files and annot_files must match.')
        
        self.count_files = count_files
        self.annot_files = annot_files
        self.select_genes = select_genes

        self.h_st = h_st
        self.w_st = w_st
        self.Visium = Visium

        self.cfile_delim = cfile_delim
        self.afile_delim = afile_delim
    
    def __len__(self):
        return len(self.count_files)
    
    def __getitem__(self, idx):
        counts_grid, annots_grid, _, _ = read_annotated_starray(self.count_files[idx], self.annot_files[idx], 
            select_genes=self.select_genes, h_st=self.h_st, w_st=self.w_st, Visium=self.Visium,
            cfile_delim=self.cfile_delim, afile_delim=self.afile_delim)

        counts_grid = torch.from_numpy(counts_grid)
        counts_grid = counts_grid.permute(2, 0, 1)  # Reshape to channels-first ordering expected by PyTorch
        
        annots_grid = torch.from_numpy(annots_grid)
        
        return counts_grid.float(), annots_grid.long()


############### Helper functions ###############

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
def read_annotated_starray(count_file, annot_file, select_genes=None, 
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
    
    amat = pd.read_csv(annot_file, header=0, index_col=0, sep=afile_delim)
    
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
        if cstr in amat.columns and np.sum(amat[cstr].values) > 0:
            counts_grid[y, x] = cmat[cstr].values
            annots_grid[y, x] = np.argmax(amat[cstr].values) + 1
    
    return counts_grid, annots_grid, cmat.index.values, amat.index.values
