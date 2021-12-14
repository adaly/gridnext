import os
import re
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from count_datasets import pseudo_hex_to_oddr


class PatchDataset(Dataset):
	def __init__(self, img_files, annot_files, img_transforms=None, afile_delim='\t', verbose=False):
		super(PatchDataset, self).__init__()

		if len(img_files) != len(annot_files):
			raise ValueError('Length of img_files and annot_files must match.')

		self.imgpath_mapping = []
		self.annotations = []

		self.afile_delim = afile_delim

		bad_annots = 0
		missing_img = 0

		# Find all annotated patches with image data
		for (imdir, afile) in zip(img_files, annot_files):
			with open(afile, 'r') as fh:				
				adat = pd.read_csv(afile, header=0, index_col=0, sep=self.afile_delim)
				for cstr in adat.columns:
					
					# Skip over unannotated or mis-annotated spots
					if not np.sum(adat[cstr]) == 1:
						if verbose:
							print(afile, cstr, 'improper annotation')
						bad_annots += 1
						continue

					# Skip over spots without image data
					imgpath = os.path.join(imdir, cstr + '.jpg')
					if not os.path.exists(imgpath):
						if verbose:
							print(imdir, cstr, 'no image data')
						missing_img += 1
						continue

					self.annotations.append(np.argmax(adat[cstr]))
					self.imgpath_mapping.append(imgpath)

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

		print('%d mis/un-annotated spots, %d spots missing image data' % (bad_annots, missing_img))

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# Read in image and apply relevant preprocessing transforms
		img = Image.open(self.imgpath_mapping[idx])
		img = self.preprocess(img)

		label = torch.tensor(self.annotations[idx]).long()
		return img.float(), label


class PatchGridDataset(Dataset):
	def __init__(self, img_files, annot_files, h_st=78, w_st=64, Visium=True,
		img_transforms=None, afile_delim='\t'):

		if len(img_files) != len(annot_files):
			raise ValueError('Length of img_files and annot_files must match.')

		self.img_files = img_files
		self.annot_files = annot_files
		self.h_st = h_st
		self.w_st = w_st
		self.Visium = Visium
		self.afile_delim = afile_delim

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __len__(self):
		return len(self.annot_files)

	def __getitem__(self, idx):
		adat = pd.read_csv(self.annot_files[idx], header=0, index_col=0, sep=self.afile_delim)

		patch_grid = None
		annots_grid = torch.zeros((self.h_st, self.w_st), dtype=int)

		rxp = re.compile("(\d+)_(\d+).jpg")
		for f in os.listdir(str(self.img_files[idx])):
			res = rxp.match(f)
			if res is not None:
				x, y = int(res.groups()[0]), int(res.groups()[1])

				# All spots must also be annotated
				cstr = '%d_%d' % (x, y)
				if cstr not in adat.columns or adat[cstr].sum() != 1:
					continue

				patch = Image.open(os.path.join(self.img_files[idx], f))
				patch = self.preprocess(patch)

				if patch_grid is None:
					c,h,w = patch.shape
					patch_grid = torch.zeros(self.h_st, self.w_st, c, h, w)
				
				if self.Visium:
					x, y = pseudo_hex_to_oddr(x, y)

				patch_grid[y, x] = patch
				annots_grid[y, x] = np.argmax(adat[cstr])

		return patch_grid.float(), annots_grid.long()
