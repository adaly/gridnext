# GridNext

Convolutional neural network architecture for supervised registration of barcoded spatial transcriptomics (ST) data. 

Extending the work of [GridNet](https://github.com/flatironinstitute/st_gridnet), **GridNext** provides the ability to learn over image data, count data, or both together. As with GridNet, the basic structure is a two-layer convolutional neural network:
1. *f* -- spot classifier applied independently to each measurement
2. *g* -- convolutional correction network applied on the output of *f* incorporating spatial information at a bandwidth dictated by network depth/kernel width.

The GridNext library additionally provides functionality for interfacing directly with data from 10x Genomics' Visium platform (through the outputs of [Spaceranger](https://www.10xgenomics.com/support/software/space-ranger/analysis/outputs/output-overview) and [Loupe](https://www.10xgenomics.com/support/software/loupe-browser/tutorials/introduction/lb-navigation-for-spatial) for count and annotation data, respectively), as well as guidelines for working with custom data in popular formats such as AnnData.

## Installation

GridNext can be installed by pip:

```
pip install git+https://github.com/adaly/gridnext/
```

Or the source code can manually be downloaded and compiled:

```
git clone https://github.com/adaly/gridnext.git
cd gridnext
pip install .
```

## Importing data

The GridNext library provides extensions of the PyTorch `Dataset` class for working with ST data:
- `CountDataset`, `PatchDataset`, and `MultiModalDataset` for spot-wise (pre) training of the *f* network
- `CountGridDataset`, `PatchGridDataset` and `MultiModalGridDataset` for training/fine-tuning the *g* network

### Visium data

The `create_visium_dataset` function is used to instatiate either spot- or grid-level datasets using either or both of the count or image data modalities from the outputs of [Spaceranger](https://www.10xgenomics.com/support/software/space-ranger/analysis/outputs/output-overview) (count and image data) and [Loupe](https://www.10xgenomics.com/support/software/loupe-browser/tutorials/introduction/lb-navigation-for-spatial) (spot annotations):

```
import os, glob
from gridnext.visium_datasets import create_visium_dataset

data_dir = '../data/BA44_testdata'

# Top-level output directories from Spaceranger for each Visium array
spaceranger_dirs = sorted(glob.glob(os.path.join(data_dir, 'spaceranger', '*')))

# Associated full-resolution image files used as inputs to Spaceranger
fullres_image_files = sorted(glob.glob(os.path.join(data_dir, 'fullres_images', '*.jpg')))

# Spot annotation files produced using the Loupe browser
annot_files = sorted(glob.glob(os.path.join(data_dir, 'annotations', '*.csv')))
```

For **count data**:
```
# Spot-wise data (for training f)
spot_dat = create_visium_dataset(spaceranger_dirs, use_count=True, use_image=False, annot_files=annot_files, spatial=False)
len(spot_dat)                  # n_spots
x_spot, y_spot = spot_dat[0]   # x_spot = (n_genes,), y_spot = (,)

# Grid-wise data (for training g)
grid_dat = create_visium_dataset(spaceranger_dirs, use_count=True, use_image=False, annot_files=annot_files, spatial=True)
len(grid_dat)                  # n_arrays
x_grid, y_grid = grid_dat[0]   # x_grid = (n_genes, n_rows_vis, n_cols_vis), y_grid = (n_rows_vis, n_cols_vis)
```

The first time this is run for a given dataset, it will create a unified list of n_genes seen across all Visium arrays, and save a unified TSV-formatted (n_genes x n_spots) count file with the suffix `.unified.tsv.gz` (can be changed with the `count_suffix` optional argument) in the top-level directory of each Spaceranger output. On subsequent runs, the function will look for these unified count files and use them in constructing count tensors. 

Optional arguments:
- `minimum_detection_rate` -- after generating unified gene list, drop genes expressed in fewer than this fraction of spots
- `select_genes` -- list of gene names (ENSEMBL if using default 10x reference transcriptome) to subset for analysis

Alternatively, if you are working with **only count data**, it may be faster to store your Visium data in an AnnData object using `create_visium_anndata`:
```
from gridnext.visium_datasets import create_visium_anndata

adata = create_visium_anndata(spaceranger_dirs, annot_files=annot_files, destfile=PATH_TO_SAVE_ANNDATA)
```
and then follow the [instructions for loading AnnData objects](#anndata-objects).

For **image data**:
```
patch_size = 128

# Spot-wise data (for training f)
spot_dat = create_visium_dataset(spaceranger_dirs, use_image=True, use_count=False, annot_files=annot_files, fullres_image_files=fullres_image_files, patch_size=patch_size, spatial=False)
len(spot_dat)                  # n_spots
x_spot, y_spot = spot_dat[0]   # x_spot = (3, patch_size, patch_size)

# Grid-wise data (for training g)
grid_dat = create_visium_dataset(spaceranger_dirs, use_image=True, use_count=False, annot_files=annot_files, fullres_image_files=fullres_image_files, patch_size=patch_size, spatial=True)
len(grid_dat)                  # n_arrays
x_grid, y_grid = grid_dat[0]   # x_grid = (n_rows_vis, n_cols_vis, 3, patch_size, patch_size)
```

The first time this runs for a given dataset, it will create a sub-directory in each spaceranger output directory with the suffix `*_patches[patch_size]` containing image patches extracted from each spot location in the array (named as [array_name]_[array_col]_[array_row].jpg). On subsequent runs with the same patch size, the function will look for these patches and use them in constructing image tensors.

Optional arguments:
- `img_transforms` -- a `torchvision.transforms` object (or a [composition](https://pytorch.org/vision/0.9/transforms.html) thereof) to be applied to any image patch prior to accession through the Dataset class.

### AnnData objects

GridNext provides functionality to load spatially resolved count data into memory from [AnnData](https://anndata.readthedocs.io/en/latest/). The AnnData object must be structured as such:
- For spatially-resolved data, `adata.obs` **must** have columns named `x` and `y` for spatial coordinate data.
- Principal components of count data may be stored in `adata.layers['X_pca']

There are two methods of loading such data:

1. For **smaller** datasets (that can be loaded into memory at once), the following functions yield `TensorDataset` objects with fast accession times:
```
import scanpy as sc
from gridnext.count_datasets import anndata_to_tensordataset, anndata_arrays_to_tensordataset

adata = sc.read_h5ad(...)  # load AnnData object
obs_label = 'AARs'         # column in adata.obs containing spot annotations
obs_arr = 'vis_arr'        # column in adata.obs.containing array names (for multi-array ST data)
h_st, w_st = 78, 64        # height and width of ST array (i.e., number of rows and columns)
vis_coords = True          # whether x and y coordinates are in Visium pseudo-hex (True) or cartesian coordinates (False)
use_pcs = False            # whether to use principal components (adata.layers['X_pca']) instead of (raw) count data (adata.X)

spot_dat = anndata_to_tensordata(adata, obs_label=obs_label, use_pcs=use_pcs)
grid_dat = anndata_arrays_to_tensordata(adata, obs_label=obs_label, use_pcs=use_pcs, obs_arr=obs_arr, h_st=h_st, w_st=w_st, vis_coords=vis_coords)
```

2. For **larger** datasets that can't fit into memory, we provide subclasses of PyTorch `Dataset` objects with lazy loading (slower accession times):
```
from gridnext.count_datasets import AnnDataset, AnnGridDataset

spot_dat = AnnDataset(adata, obs_label=obs_label, use_pcs=use_pcs)
grid_dat = AnnGridDataset(adata, obs_label=obs_label, use_pcs=use_pcs, obs_arr=obs_arr, h_st=h_st, w_st=w_st, vis_coords=vis_coords)
```

### Custom data

The aforementioned dataset classes (`CountDataset`/`CountGridDataset`, `PatchDataset`/`PatchGridDataset`, `MultiModalDataset`/`MultiModalGridDataset`) can be instantiated directly by providing count, image, and annotation data in the aforementioned file formats:
- **Count data**: one (genes x spots) matrix per array, stored in tab-delimited format (other delimiters can be used with `cfile_delim` keyword argument). First column should store gene names, which should be standardized across all arrays, and first column should store spot coordinates in `[x]_[y]` format.
- **Image data**: one directory per array containing JPEG-formatted image files (other file formats can be used with `img_ext` keyword argument) extracted from each spatial measurement location. Image patch file names should end with `[x]_[y].[img_ext]` to store spatial information.
- **Annotation data**: one (categories x spots) one-hot encoded (exactly one "1" per row) binary annotation matrix per array, stored in CSV format (other delimiters can be used with `afile_delim` keyword argument). First column should store category names, which should be standardized across all arrays, and first column should store spot coordinates in `[x]_[y]` format. If Visium data are being passed (`Visium=True`), one can alternately pass paired lists of Loupe annotation files and Spaceranger position files in lieu of this custom format.

```
from gridnext.count_datasets import CountDataset, CountGridDataset
from gridnext.image_datasets import PatchDataset, PatchGridDataset
from gridnext.multimodal_datasets import MultiModalDataset, MultiModalGridDataset
```

## Model instantiation

GridNext requires two model instantiations:
1. Spot classifier *f*, which accepts either a transcriptome (1D), image patch (3D), or tuple containing both, and outputs a `n_classes`-length logit tensor
   - For image data, we provide the `gridnext.densenet.DenseNet` class for instantiating a DenseNet classifier (see [image tutorial Section 1](notebooks/Tutorial_visium_image.ipynb) for example).
   - For count data, a custom `torch.nn.Sequential` network with appropriate input and output dimensions should be used (see [count tutorial Section 1.2](notebooks/Tutorial_visium_count.ipynb) for example).
3. Grid classifier *g*, which accepts either a transcriptomic array tensor (3D), image array tensor (5D), or tuple containig both, and outputs an `(n_classes, H_ST, W_ST)`-shaped logit tensor
   - For Visium data (either count or image), we provide the `gridnext.gridnet_models.GridNetHexOddr` class. In either instance (see [image tutorial Section 2](notebooks/Tutorial_visium_image.ipynb) or [count tutorial Section 2.2](notebooks/Tutorial_visium_count.ipynb)), instantiation requires:
     - (pre-trained) patch classifier
     - shape of spot data
     - shape of spatial grid
     - number of classes in final prediction layer
   - By default, *g* takes as input the final output layer of *f* (spot_shape -> f -> n_classes -> g -> n_classes). To instead learn over a penulatimate feature layer of *f*, create a truncated network (e.g., a `DenseNet` model instantiated with the `classify=False` option) and instantiate `GridNetHexOddr` with the `f_dim=SHAPE_OF_F_OUTPUT` option.
   - For data too large to fit into RAM at once (e.g., tensors of image data), `GridNetHexOddr` provides the `atonce_patch_limit` instantiation argument, which splits arrays into mini-batches during training (see example in [image tutorial Section 2](notebooks/Tutorial_visium_image.ipynb)).

## Model training

GridNext provides two functions for model training:
1. `gridnext.training.train_spotwise` for training *f* networks
2. `gridnext.training.train_gridwise` for training *g* networks

Both functions require the following arguments:
- `model`: either *f* or *g*
- `dataloaders`: dictionary mapping keys "train" and "val" to separate `torch.utils.data.DataLoader` objects for each data fold
- `criterion`: loss function from `torch.nn`
- `optimizer`: optimizer from `torch.optim`

Both functions additionally accept the following optional arguments:
- `num_epochs`: number of training epochs (defaults to 10)
- `outfile`: destination in which to save trained model parameters (updated each iteration)

See examples in [either](notebooks/Tutorial_visium_image.ipynb) [tutorial](notebooks/Tutorial_visium_count.ipynb)

## Output visualization
