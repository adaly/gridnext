import os
import re
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from gridnext.count_datasets import CountDataset, CountGridDataset
from gridnext.count_datasets import AnnDataset, AnnGridDataset
from gridnext.imgprocess import pseudo_hex_to_oddr


# Accepts pair of image, count datasets of identical length and output dimension
# - Zeroes out entries in output for which two datasets do not agree
class MMStackDataset(Dataset):
	def __init__(self, image_dataset, count_dataset):
		assert len(count_dataset) == len(image_dataset), "Datasets must be of the same length!"
		self.count_dataset = count_dataset
		self.image_dataset = image_dataset

	def __len__(self):
		return len(self.count_dataset)

	def __getitem__(self, idx):
		(x1,y1), (x2,y2) = self.image_dataset[idx], self.count_dataset[idx]

		diff = y1 != y2
		y = torch.clone(y1)
		y[diff] = 0

		return (x1, x2), y


# Accepts AnnData object constructed with visium_datasets.create_visium_anndata_img
class MMAnnDataset(AnnDataset):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing count data (in X/obsm) and image data (in obs) from ST arrays
	obs_label: str
		column in adata.obs containing the spot labels to predict
	obs_img: str
		column in adata.obs containing paths to individual spot images
	use_pcs: int or None
		number of PCs (from adata.obsm['X_pca']) to use as input, or None to use adata.X
	img_transforms: torchvision.Transform
		preprocessing transforms to apply to image patches after loading
	'''
	def __init__(self, adata, obs_label, obs_img='imgpath', use_pcs=None, img_transforms=None):
		super(MMAnnDataset, self).__init__(adata, obs_label, use_pcs=use_pcs)

		self.imgfiles = adata.obs[obs_img]

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __getitem__(self, idx):
		x_count, y = super(MMAnnDataset, self).__getitem__(idx)
		x_image = Image.open(self.imgfiles[idx])
		x_image = self.preprocess(x_image).float()

		return (x_image, x_count), y

class MMAnnGridDataset(AnnGridDataset):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing count data (in X/obsm) and image data (in obs) from Visium arrays
	obs_label: str
		column in adata.obs containing the spot labels to predict
	obs_arr: str
		column in adata.obs containing the array labels for each spot
	obs_img: str
		column in adata.obs containing paths to individual spot images
	use_pcs: int or None
		number of PCs (from adata.obsm['X_pca']) to use as input, or None to use adata.X
	img_transforms: torchvision.Transform
		preprocessing transforms to apply to image patches after loading
	obs_x, obs_y: str
		column in adata.obs containing x and y ST array coordinates
	h_st, w_st: int
		number of rows, columns in ST array
	vis_coords: bool
		whether the coordinates in adata.obs.obs_x and adata.obs.obs_y are in Visium pseudo-hex format
	'''
	def __init__(self, adata, obs_label, obs_arr, obs_img='imgpath', use_pcs=None, img_transforms=None, 
		obs_x='x', obs_y='y', h_st=78, w_st=64, vis_coords=True):

		super(MMAnnGridDataset, self).__init__(adata, obs_label, obs_arr, obs_x=obs_x, obs_y=obs_y,
			h_st=h_st, w_st=w_st, use_pcs=use_pcs, vis_coords=vis_coords)

		self.obs_img = obs_img

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __getitem__(self, idx):
		x_count, y = super(MMAnnGridDataset, self).__getitem__(idx)

		adata_arr = self.adata[self.adata.obs[self.obs_arr]==self.arrays[idx]]
		patch_grid = None
		for imfile, a_x, a_y in zip(adata_arr.obs[self.obs_img], adata_arr.obs[self.obs_x], 
			adata_arr.obs[self.obs_y]):
			
			patch = Image.open(imfile)
			patch = self.preprocess(patch)

			if patch_grid is None:
				c,h,w = patch.shape
				patch_grid = torch.zeros(self.h_st, self.w_st, c, h, w)
			
			if self.vis_coords:
				x, y = pseudo_hex_to_oddr(a_x, a_y)
			else:
				x, y = a_x, a_y
			patch_grid[y, x] = patch

		x_image = patch_grid.float()

		return (x_image, x_count), y


############ CURRENTLY DEFUNCT ############

# Currently expects Splotch-formatted annotation files, which are useful because one-hot representations
# define consistent class indexing across samples. 
# TODO: Should we move to all-Visium? Still have to process count data in Splotch to produce unified gene list.

class MultiModalDataset(CountDataset):
	def __init__(self, count_files, img_files, annot_files, select_genes=None, img_transforms=None,
		cfile_delim='\t', afile_delim='\t'):
		super(CountDataset, self).__init__()

		if len(count_files) != len(img_files) or len(count_files) !=  len(annot_files):
			raise ValueError('Length of count_files, img_files and annot_files must match.')

		self.select_genes = select_genes
		self.countfile_mapping = []
		self.imgpath_mapping = []
		self.cind_mapping = []
		self.annotations = []

		self.cfile_delim = cfile_delim
		self.afile_delim = afile_delim

		# Find all annotated patches with image and count data
		for (cfile, imdir, afile) in zip(count_files, img_files, annot_files):
			with open(cfile, 'r') as fh:
				counts_header = next(fh).strip('\n').split(self.cfile_delim)
				
				adat = pd.read_csv(afile, header=0, index_col=0, sep=self.afile_delim)
				for cstr in adat.columns:
					
					# Skip over unannotated or mis-annotated spots
					if not cstr in counts_header:
						print(afile, cstr, 'missing')
						continue
					counts_ind = counts_header.index(cstr)

					if not np.sum(adat[cstr]) == 1:
						print(afile, cstr, 'improper annotation')
						continue

					# Skip over spots without image data
					imgpath = os.path.join(imdir, cstr + '.jpg')
					if not os.path.exists(imgpath):
						print(imdir, cstr, 'no image data')
						continue

					self.annotations.append(np.argmax(adat[cstr]))
					self.countfile_mapping.append(cfile)
					self.imgpath_mapping.append(imgpath)
					self.cind_mapping.append(counts_ind)

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __getitem__(self, idx):
		count_vec, label = super(MultiModalDataset, self).__getitem__(idx)

		# Read in image and apply relevant preprocessing transforms
		img = Image.open(self.imgpath_mapping[idx])
		img = self.preprocess(img)

		return count_vec, img.float(), label


class MultiModalGridDataset(CountGridDataset):
	def __init__(self, count_files, img_files, annot_files, select_genes=None, h_st=78, w_st=64, Visium=True,
		img_transforms=None, cfile_delim='\t', afile_delim='\t'):
		super(MultiModalGridDataset, self).__init__(count_files, annot_files, select_genes, 
			h_st, w_st, Visium, cfile_delim, afile_delim)

		self.img_files = img_files

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __getitem__(self, idx):
		counts_grid, annots_grid = super(MultiModalGridDataset, self).__getitem__(idx)

		patch_grid = None

		rxp = re.compile("(\d+)_(\d+).jpg")
		for f in os.listdir(self.img_files[idx]):
			res = rxp.match(f)
			if res is not None:
				x, y = int(res.groups()[0]), int(res.groups()[1])

				patch = Image.open(os.path.join(self.img_files[idx], f))
				patch = self.preprocess(patch)

				if patch_grid is None:
					c,h,w = patch.shape
					patch_grid = torch.zeros(self.h_st, self.w_st, c, h, w)
				
				if self.Visium:
					x, y = pseudo_hex_to_oddr(x, y)
				patch_grid[y, x] = patch

		# Spots must have image data and annotations to be marked as foreground.
		for i in range(self.h_st):
			for j in range(self.w_st):
				if patch_grid[i,j].max() == 0:
					annots_grid[i,j] = 0
					counts_grid[:,i,j] = 0
				if annots_grid[i,j] == 0:
					patch_grid[i,j] = 0

		return counts_grid, patch_grid.float(), annots_grid

