# GridNext

Convolutional neural network architecture for supervised registration of barcoded spatial transcriptomics (ST) data. 

Extending the work of [GridNet](https://github.com/flatironinstitute/st_gridnet), **GridNext** provides the ability to learn over image data, count data, or both together. As with GridNet, the basic structure is a two-layer convolutional neural network:
1. *f* -- spot classifier applied independently to each measurement
2. *g* -- convolutional correction network applied on the output of *f* incorporating spatial information at a bandwidth dictated by network depth/kernel width.

The GridNext library additionally provides functionality for interfacing directly with data from 10x Genomics' Visium platform (through the outputs of [Spaceranger](https://www.10xgenomics.com/support/software/space-ranger/analysis/outputs/output-overview) and [Loupe](https://www.10xgenomics.com/support/software/loupe-browser/tutorials/introduction/lb-navigation-for-spatial) for count and annotation data, respectively), as well as guidelines for working with custom data in popular formats such as HDF5-formatted AnnData.

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

For **image data**
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

### AnnData (count only)

### Custom data

The aforementioned dataset classes can be instantiated directly by providing count, image, and annotation data in the aforementioned file formats:
