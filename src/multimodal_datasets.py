import os
import re
import time
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from count_datasets import CountDataset, CountGridDataset
from imgprocess import pseudo_hex_to_oddr


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


if __name__ == '__main__':
	train_tissues = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674']

	data_dir = '/Users/adaly/Documents/Splotch_projects/Maynard_DLPFC/data/'
	countfiles_train = [os.path.join(data_dir, 'Countfiles_Visium_norm/%s_stdata_aligned_counts_IDs.txt.unified.tsv') % s for s in train_tissues]
	annotfiles_train = [os.path.join(data_dir, 'Covariates_Visium/%s.tsv') % s for s in train_tissues]
	imgfiles_train = [os.path.join(data_dir, 'maynard_patchdata_oddr/%s_full_image/') % s for s in train_tissues]

	# Joana's manually curated list of layer marker genes:
	jp_markers = {
		'MBP': 'ENSG00000197971',    # WM
		'SNAP25': 'ENSG00000132639', # GM (Layers 1-6)
		'PCP4': 'ENSG00000183036',   # Layer 5
		'RORB': 'ENSG00000198963',   # Layer 4
		'SYNPR': 'ENSG00000163630',  # Layer 6
		'MFGE8': 'ENSG00000140545',
		'CBLN2': 'ENSG00000141668',
		'RPRM': 'ENSG00000177519',
		'NR4A2': 'ENSG00000153234',
		'CXCL14': 'ENSG00000145824',
		'C1QL2': 'ENSG00000144119',
		'CUX2': 'ENSG00000111249',
		'CARTPT': 'ENSG00000164326',
		'CCK': 'ENSG00000187094'
	}
	select_genes = [ensmbl for _, ensmbl in jp_markers.items()]

	#ds = MultiModalDataset(countfiles_train, imgfiles_train, annotfiles_train, select_genes=select_genes)
	
	ds2 = MultiModalGridDataset(countfiles_train, imgfiles_train, annotfiles_train, select_genes=select_genes)

	start = time.time()
	cmat, imat, amat = ds2[9]
	print(cmat.shape, cmat.min(), cmat.max())
	print(imat.shape, imat.min(), imat.max())
	print(amat)
	print(time.time()-start)
