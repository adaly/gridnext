# GridNext

Convolutional neural network architecture for supervised registration of barcoded spatial transcriptomics (ST) data. 

Extending the work of [GridNet](https://github.com/flatironinstitute/st_gridnet), **GridNext** provides the ability to learn over image data, count data, or both together. As with GridNet, the basic structure is a two-layer convolutional neural network:
1. *f* -- spot classifier applied independently to each measurement
2. *g* -- convolutional correction network applied on the output of *f* incorporating spatial information at a bandwidth dictated by network depth/kernel width.

The GridNext library additionally provides functionality for interfacing directly with data from 10x Genomics' Visium platform (through the outputs of Spaceranger and Loupe for count and annotation data, respectively), as well as guidelines for working with custom data in popular formats such as HDF5-formatted AnnData.

## Importing data

The GridNext library provides extensions of the PyTorch `Dataset` class for working with ST data:
- `CountDataset`, `PatchDataset`, and `MultiModalDataset` for spot-wise (pre) training of the *f* network
- `CountGridDataset`, `PatchGridDataset` and `MultiModalGridDataset` for training/fine-tuning the *g* network

### Visium data

### AnnData (count only)

### Custom data

The aforementioned dataset classes can be instantiated directly by providing count, image, and annotation data in the aforementioned file formats:
