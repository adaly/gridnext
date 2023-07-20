import re
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from scipy.sparse import issparse
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import LabelEncoder
from utils import pseudo_hex_to_oddr, read_annotated_starray, anndata_to_grids


############### Read full datasets into memory ###############

def load_count_dataset(count_files, annot_files=None, select_genes=None):
    count_data_full = []
    annot_data_full = []

    for i, cf in enumerate(count_files):
        if annot_files is not None:
            adat = pd.read_csv(annot_files[i], header=0, index_col=0, sep='\t')

            # Remove all unannotated or mis-annotated columns
            for cstr in adat.columns:
                if adat[cstr].values.sum() != 1:
                    adat.drop(columns=cstr, inplace=True)
        else:
            adat = None

        cdat = pd.read_csv(cf, header=0, index_col=0, sep='\t')

        for cstr in cdat.columns:
            if adat is not None and cstr not in adat.columns:
                continue

            if select_genes is not None:
                count_data_full.append(cdat.loc[select_genes,cstr].values.squeeze())
            else:
                count_data_full.append(cdat[cstr].values.squeeze())

            if adat is not None:
                annot_data_full.append(adat[cstr].values.argmax())
            else:
                annot_data_full.append(0)

    count_data_full = torch.tensor(np.array(count_data_full)).float()
    annot_data_full = torch.tensor(annot_data_full).long()
    
    return TensorDataset(count_data_full, annot_data_full)

def load_count_grid_dataset(count_files, annot_files=None, select_genes=None, 
    h_st=78, w_st=64, Visium=True):
    count_data_full = []
    annot_data_full = []
    
    for i, cf, in enumerate(count_files):
        if annot_files is not None:
            af = annot_files[i]
        else:
            af = None

        counts_grid, annots_grid, _, _ = read_annotated_starray(cf, af, select_genes=select_genes,
            h_st=h_st, w_st=w_st, Visium=Visium)
        
        count_data_full.append(counts_grid)
        annot_data_full.append(annots_grid)
    
    count_data_full = torch.tensor(np.array(count_data_full)).permute(0,3,1,2)  # Reshape to channels-first
    annot_data_full = torch.tensor(np.array(annot_data_full))
    
    return TensorDataset(count_data_full.float(), annot_data_full.long())


############### Map-style PyTorch datasets ###############

class CountDataset(Dataset):
    # For independent classification of spots based on 1d expression vectors.

    def __init__(self, count_files, annot_files=None, select_genes=None,
        cfile_delim='\t', afile_delim='\t', verbose=False):
        super(CountDataset, self).__init__()
                
        if annot_files is not None and not len(count_files) == len(annot_files):
            raise ValueError('Length of count_files and annot_files must match.')

        self.cfile_delim = '\t'
        self.afile_delim = '\t'
        
        self.countfile_mapping = []
        self.annotations = []
        self.cind_mapping = []

        bad_annots = 0
        missing_annots = 0
        rxp_cstr = re.compile('\d+_\d+')
        
        # Construct unique integer index for all annotated patches
        for i, cf in enumerate(count_files):
            with open(cf, 'r') as fh:
                counts_header = next(fh).strip('\n').split(self.cfile_delim)

                if annot_files is not None:
                    af = annot_files[i]
                    adat = pd.read_csv(af, header=0, index_col=0, sep=self.afile_delim)

                    for cstr in adat.columns:
                        # Skip over unannotated or mis-annotated spots
                        if not cstr in counts_header or np.sum(adat[cstr])==0:
                            if verbose:
                                print(af, cstr, 'missing')
                            missing_annots += 1
                            continue

                        counts_ind = counts_header.index(cstr)

                        if not np.sum(adat[cstr]) == 1:
                            if verbose:
                                print(af, cstr, 'improper annotation')
                            bad_annots += 1
                            continue

                        self.annotations.append(np.argmax(adat[cstr]))
                        self.countfile_mapping.append(cf)
                        self.cind_mapping.append(counts_ind)
                else:
                    for counts_ind, cstr in enumerate(counts_header):
                        if rxp_cstr.match(cstr) is not None:
                            self.countfile_mapping.append(cf)
                            self.cind_mapping.append(counts_ind+1)
        
        self.select_genes = select_genes

        if annot_files is not None:
            print('%d un-annotated spots, %d mis-annotated spots' % (missing_annots, bad_annots))
        
    def __len__(self):
        return len(self.cind_mapping)

    def __getitem__(self, idx):
        if self.select_genes is not None:
            count_vec = self._get_select(idx)
        else:
            df = pd.read_csv(self.countfile_mapping[idx], sep=self.cfile_delim, header=0, index_col=0,
                            usecols=[0, self.cind_mapping[idx]])
            count_vec = torch.from_numpy(df.values.squeeze())

        if len(self.annotations) > 0:
            label = torch.tensor(self.annotations[idx]).long()
        else:
            label = torch.tensor(0).long()

        return count_vec.float(), label
    
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
            
        return torch.tensor(count_vec).float()

class CountGridDataset(Dataset):
    # For registration of entire ST arrays based on 3d expression maps.

    def __init__(self, count_files, annot_files=None, select_genes=None, 
        h_st=78, w_st=64, Visium=True, cfile_delim='\t', afile_delim='\t'):
        super(CountGridDataset, self).__init__()
        
        if annot_files is not None and not len(count_files) == len(annot_files):
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
        if self.annot_files is not None:
            af = self.annot_files[idx]
        else:
            af = None

        counts_grid, annots_grid, _, _ = read_annotated_starray(self.count_files[idx], af, 
            select_genes=self.select_genes, h_st=self.h_st, w_st=self.w_st, Visium=self.Visium,
            cfile_delim=self.cfile_delim, afile_delim=self.afile_delim)

        counts_grid = torch.from_numpy(counts_grid)
        counts_grid = counts_grid.permute(2, 0, 1)  # Reshape to channels-first ordering expected by PyTorch
        
        annots_grid = torch.from_numpy(annots_grid)
        
        return counts_grid.float(), annots_grid.long()


############### AnnData-based Datasets ###############

class AnnDataset(Dataset):
    def __init__(self, adata, obs_label, use_pcs=False):
        super(AnnDataset, self).__init__()
        self.adata = adata
        self.use_pcs = use_pcs
        self.obs_label = obs_label

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(adata.obs[obs_label])
    
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        y = self.labels[idx]
        if self.use_pcs is not None:
            x = self.adata.obsm['X_pca'][idx, :self.use_pcs]
        else:
            x = self.adata.X[idx, :]
        
        return torch.from_numpy(x), torch.tensor(y).long()


# Return a TensorDataset matching spot count data (either from X or X_pca) to obs_label data from adata.obs.
# (MUCH faster than AnnDataset)
def anndata_to_tensordataset(adata, obs_label, use_pcs=False):
    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[obs_label])
    print(le.classes_)
    
    if use_pcs:
        count_data = adata.obsm['X_pca'][:, :use_pcs]
    else:
        count_data = adata.X

    if issparse(count_data):
        count_data = count_data.todense()
        
    return TensorDataset(torch.tensor(count_data).float(),
                         torch.tensor(labels).long())


# Subset AnnData object by obs_arr (e.g., Visium array ID)
# -> FAST instantiation (<1s), SLOW accession (~20s/array)
class AnnGridDataset(AnnDataset):
    def __init__(self, adata, obs_label, obs_arr, h_st=78, w_st=64, use_pcs=False, 
                 vis_coords=True):
        super(AnnGridDataset, self).__init__(adata, obs_label, use_pcs)
        
        self.h_st = h_st
        self.w_st = w_st
        self.obs_arr = obs_arr
        self.vis_coords = vis_coords
        
        self.arrays = adata.obs[obs_arr].unique()
    
    def __len__(self):
        return len(self.adata.obs[self.obs_arr].unique())
    
    def __getitem__(self, idx):
        adata_arr = self.adata[self.adata.obs[self.obs_arr]==self.arrays[idx]]
        lbls_arr = self.le.transform(adata_arr.obs[self.obs_label].values)
        
        counts_grid, labels_grid = anndata_to_grids(adata_arr, lbls_arr, self.h_st, 
                                                    self.w_st, self.use_pcs, self.vis_coords)
        return counts_grid.float(), labels_grid.long()

    
# Load full datset into memory as TensorDataset -- slow instantiation
# -> SLOW instantiation (~20s/array), FAST accession (<1s)
def anndata_arrays_to_tensordataset(adata, obs_label, obs_arr, h_st=78, w_st=64, 
                                    use_pcs=False, vis_coords=True):
    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[obs_label])
    
    count_grids, label_grids = [],[]
    
    for arr in tqdm(adata.obs[obs_arr].unique()):
        adata_arr = adata[adata.obs[obs_arr]==arr]
        lbls_arr = le.transform(adata_arr.obs[obs_label].values)
        
        cg, lg = anndata_to_grids(adata_arr, lbls_arr, h_st, w_st, use_pcs, vis_coords)
        count_grids.append(cg)
        label_grids.append(lg)
            
    return TensorDataset(torch.stack(count_grids), torch.stack(label_grids))
