import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from count_datasets import pseudo_hex_to_oddr
from utils import read_annotfile


class PatchDataset(Dataset):
	def __init__(self, img_files, annot_files=None, position_files=None, Visium=True,
		img_transforms=None, afile_delim=',', img_ext='jpg', verbose=False):
		'''
		Parameters:
		----------
		img_files: iterable of path
			one sub-directory per ST array, each containing spot images named as "*_[array_xcoord]_[array_ycoord].[img_ext]"
		annot_files: iterable of path
			one annotation file per ST array, in either:
			- Loupe (Visium) format: barcode, annotation columns
			- classic ST format: array_coords x annotations binary matrix
		position_files: iterable of path
			for Visium data, tissue position file output by Spaceranger mapping spatial barcodes to array/pixel coordinates
		Visium: bool
			Visium data (default) or classic ST data
		img_transforms: Transform
			composition of torchvision Transforms to be applied to input images
		afile_delim: char
			delimiter for annotation file
		img_ext: str
			extension used for image data (default: .jpg)
		verbose: bool
			print details of missing patches/annotations
		'''

		super(PatchDataset, self).__init__()

		if annot_files is not None and len(img_files) != len(annot_files):
			raise ValueError('Length of img_files and annot_files must match.')

		if Visium:
			if annot_files is not None:
				if position_files is None:
					raise ValueError('Must provide Spaceranger position files mapping barcodes to array locations.')
				if len(position_files) != len(annot_files):
					raise ValueError('Number of Spaceranger position files does not match number of annotation files.')

				# Map set of all unique annotations to integer values
				all_annots = np.array([])
				for afile, pfile in zip(annot_files, position_files):
					_, annot_strs = read_annotfile(afile, position_file=pfile, Visium=True)
					all_annots = np.union1d(all_annots, annot_strs)

				le = LabelEncoder()
				le.fit(all_annots)
				self.classes = le.classes_

		self.imgpath_mapping = []
		self.annotations = []

		self.afile_delim = afile_delim

		bad_annots = 0
		missing_img = 0

		# Find all annotated patches with image data
		if annot_files is not None:
			for i, (imdir, afile) in enumerate(zip(img_files, annot_files)):
				if Visium:
					coord_strs, annot_strs = read_annotfile(afile, position_file=position_files[i], 
						Visium=True, afile_delim=self.afile_delim)
					annot_lbls = le.transform(annot_strs)
				else:	
					coord_strs, annot_lbls = read_annotfile(afile, Visium=False, afile_delim=self.afile_delim)

				adict = dict(zip(coord_strs, annot_lbls))

				imgpaths = glob.glob(os.path.join(imdir, '*.' + img_ext))
				for imfile in imgpaths:
					cstr = '_'.join(Path(imfile).stem.split('_')[-2:])

					if cstr not in adict.keys():
						if verbose:
							print(cstr, 'image patch missing annotation (skipping)')
						continue

					self.annotations.append(adict[cstr])
					self.imgpath_mapping.append(imfile)
		else:
			self.imgpath_mapping = np.concatenate([glob.glob(os.path.join(imdir, '*.' + img_ext)) for imdir in img_files])

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

		if annot_files is not None and verbose:
			print('%d mis/un-annotated spots, %d spots missing image data' % (bad_annots, missing_img))

	def __len__(self):
		return len(self.imgpath_mapping)

	def __getitem__(self, idx):
		# Read in image and apply relevant preprocessing transforms
		img = Image.open(self.imgpath_mapping[idx])
		img = self.preprocess(img)

		if len(self.annotations) > 0:
			label = torch.tensor(self.annotations[idx]).long()
		else:
			label = torch.empty(0)
		return img.float(), label


class PatchGridDataset(Dataset):
	def __init__(self, img_files, annot_files=None, position_files=None, Visium=True, 
		img_transforms=None, afile_delim=',', img_ext='jpg', h_st=78, w_st=64):
		'''
		Parameters:
		----------
		img_files: iterable of path
			one sub-directory per ST array, each containing spot images named as "*_[array_xcoord]_[array_ycoord].[img_ext]"
		annot_files: iterable of path
			one annotation file per ST array, in either:
			- Loupe (Visium) format: barcode, annotation columns
			- classic ST format: array_coords x annotations binary matrix
		position_files: iterable of path
			for Visium data, tissue position file output by Spaceranger mapping spatial barcodes to array/pixel coordinates
		Visium: bool
			Visium data (default) or classic ST data
		img_transforms: Transform
			composition of torchvision Transforms to be applied to input images
		afile_delim: char
			delimiter for annotation file
		img_ext: str
			extension used for image data (default: .jpg)
		h_st: int
			number of rows in ST array
		w_st: int
			number of columns in ST array
		'''

		super(PatchGridDataset, self).__init__()

		if annot_files is not None and len(img_files) != len(annot_files):
			raise ValueError('Length of img_files and annot_files must match.')

		if Visium:
			if annot_files is not None:
				if position_files is None:
					raise ValueError('Must provide Spaceranger position files mapping barcodes to array locations.')
				if len(position_files) != len(annot_files):
					raise ValueError('Number of Spaceranger position files does not match number of annotation files.')

				# Map set of all unique annotations to integer values
				all_annots = np.array([])
				for afile, pfile in zip(annot_files, position_files):
					_, annot_strs = read_annotfile(afile, position_file=pfile, Visium=True)
					all_annots = np.union1d(all_annots, annot_strs)

				self.le = LabelEncoder()
				self.le.fit(all_annots)
				self.classes = self.le.classes_
				self.position_files = position_files

		self.img_files = img_files
		self.annot_files = annot_files
		self.h_st = h_st
		self.w_st = w_st
		self.Visium = Visium
		self.afile_delim = afile_delim
		self.img_ext = img_ext

		if img_transforms is None:
			self.preprocess = Compose([ToTensor()])
		else:
			self.preprocess = img_transforms

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, idx):
		if self.annot_files is not None:
			if self.Visium:
				coord_strs, annot_strs = read_annotfile(
					self.annot_files[idx], position_file=self.position_files[idx], Visium=True, 
					afile_delim=self.afile_delim)
				annot_lbls = self.le.transform(annot_strs)
			else:	
				coord_strs, annot_lbls = read_annotfile(afile, Visium=False, 
					afile_delim=self.afile_delim)

			adict = dict(zip(coord_strs, annot_lbls))

		patch_grid = None
		annots_grid = torch.zeros((self.h_st, self.w_st), dtype=int)

		rxp = re.compile(".*_(\d+)_(\d+).%s" % self.img_ext)
		for f in os.listdir(str(self.img_files[idx])):
			res = rxp.match(f)
			if res is not None:
				a_x, a_y = int(res.groups()[0]), int(res.groups()[1])

				patch = Image.open(os.path.join(self.img_files[idx], f))
				patch = self.preprocess(patch)

				if patch_grid is None:
					c,h,w = patch.shape
					patch_grid = torch.zeros(self.h_st, self.w_st, c, h, w)
				
				if self.Visium:
					x, y = pseudo_hex_to_oddr(a_x, a_y)
				else:
					x, y = a_x, a_y

				if self.annot_files is not None:
					cstr = '%d_%d' % (a_x, a_y)
					if cstr in adict.keys():
						annots_grid[y, x] = adict[cstr] + 1 # 0 reserved for background
				patch_grid[y, x] = patch

		return patch_grid.float(), annots_grid.long()
