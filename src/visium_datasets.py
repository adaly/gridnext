import os
import csv
import gzip
import glob
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

from utils import visium_get_positions, visium_find_position_file
from imgprocess import save_visium_patches, VISIUM_H_ST, VISIUM_W_ST
from image_datasets import PatchDataset, PatchGridDataset
from count_datasets import CountDataset, CountGridDataset

def create_visium_dataset(spaceranger_dirs, use_count=True, use_image=True, spatial=True,
	annot_files=None, fullres_image_files=None, count_suffix=".unified.tsv.gz", minimum_detection_rate=0.02,
	patch_size_px=128, img_transforms=None, select_genes=None):
	'''
	Parameters:
	----------
	spaceranger_dirs: iterable of path
		path to spaceranger output directories for each Visium array in dataset
	use_count: bool
		whether to employ count data
	use_image: bool
		whether to employ image data
	spatial: bool
		treat entire Visium arrays as individual inputs; otherwise learn independently over spots
	annot_files: iterable of path, or None
		path to Loupe annotation file for each array in dataset; None for un-annotated data
	fullres_image_files: iterable of path, or None
		path to full-resolution image file for each Visium array (required if use_image=True)
	count_suffix: str
		file suffix for generated unified count files
	minimum_detection_rate: float, or None
		discard genes detected in fewer than this fraction of spots across the dataset
	patch_size_px: int
		width of patches, in pixels, to be extracted at each spot location
	img_transforms: torchvision.transform
		transform to be applied to each image patch upon loading (e.g., normalization for pretrained network)
	select_genes: iterable of str
		list of genes to subset from the full transcriptome

	Returns:
	-------
	PatchDataset or PatchGridDataset or CountDataset or CountGridDataset or MultiModalDataset or
		MultiModalGridDataset
		appropriate subclass of torch.utils.data.Dataset for learning task
	'''
		
	if not (use_count or use_image):
		raise ValueError("Must utilize at least one data modality")

	# Check if unified countfiles have already been generated for these data
	if use_count:
		count_files = [os.path.join(srd, Path(srd).name+count_suffix) for srd in spaceranger_dirs]

		if not np.all([os.path.exists(cfile) for cfile in count_files]):
			print("No unified countfiles detected (%s) -- generating..." % ("*"+count_suffix))

			visium_prepare_count_files(spaceranger_dirs, count_suffix, minimum_detection_rate)

	# Check if image patches have already been extracted for these data
	if use_image:
		patch_suffix = '_patches%d' % patch_size_px
		patch_dirs = [os.path.join(srd, Path(srd).name+patch_suffix) for srd in spaceranger_dirs]

		if not np.all([os.path.exists(pdir) for pdir in patch_dirs]):
			print("No extracted image patches detected (%s) -- generating..." % ("*"+patch_suffix))

			if fullres_image_files is None:
				raise ValueError('Must provide fullres_image_files to extract image patches')

			for imfile, pdir, srd in zip(fullres_image_files, patch_dirs, spaceranger_dirs):
				if not os.path.exists(imfile):
					raise ValueError('Could not find image file: %s' % imfile)

				save_visium_patches(imfile, spaceranger_dir=srd, dest_dir=pdir, patch_size=patch_size_px)

	# Find position files mapping spot barcodes to array/pixel coordinates
	position_files = [visium_find_position_file(srd) for srd in spaceranger_dirs]

	# Count-only data
	if use_count and not use_image:
		if spatial:
			grid_data = CountGridDataset(count_files, annot_files=annot_files, position_files=position_files,
				Visium=True, select_genes=select_genes, h_st=VISIUM_H_ST, w_st=VISIUM_W_ST)
			return grid_data
		else:
			patch_data = CountDataset(count_files, annot_files=annot_files, position_files=position_files,
				Visium=True, select_genes=select_genes)
			return patch_data

	# Image-only data
	elif use_image and not use_count:
		if spatial:
			grid_data = PatchGridDataset(patch_dirs, annot_files=annot_files, position_files=position_files,
				Visium=True, img_transforms=img_transforms, h_st=VISIUM_H_ST, w_st=VISIUM_W_ST)
			return grid_data
		else:
			patch_data = PatchDataset(patch_dirs, annot_files=annot_files, position_files=position_files,
				Visium=True, img_transforms=img_transforms)
			return patch_data

	# Multimodal data
	else:
		pass



def visium_prepare_count_files(spaceranger_dirs, suffix, minimum_detection_rate=None):
	# Assemble count matrix dataframe from components in output directories:
	frames = []
	count_files = []
	for srd in spaceranger_dirs:

		matrix_dir = os.path.join(srd, "outs/filtered_feature_bc_matrix/")
		mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

		features_path = os.path.join(matrix_dir, "features.tsv.gz")
		feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, "rt"), delimiter="\t")]

		barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
		barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, "rt"), delimiter="\t")]

		positions = visium_get_positions(srd)

		positions_list = []
		for b in barcodes:
			xcoor = positions.loc[b,'array_col']
			ycoor = positions.loc[b,'array_row']
			positions_list.append('%d_%d' % (xcoor, ycoor))

		df = pd.DataFrame.sparse.from_spmatrix(mat, index=feature_ids, columns=positions_list)
		frames.append(df)
		count_files.append(os.path.join(srd, Path(srd).name))
  
	for filename,frame in zip(count_files,frames):
		frame.columns = pd.MultiIndex.from_product([[filename],frame.columns],names=['Sample','Coordinate'])
		frame.index.name = 'Gene'

	# concatenate counts
	result = pd.concat(frames,copy=False,axis=1,sort=True)
	print('We have detected %d genes across all samples'%(result.shape[0]))
	# fill NaNs with zeros
	result = result.fillna(0).astype(int)

	# discard lowly expressed genes
	if minimum_detection_rate is not None:
		result = result[((result > 0).sum(axis=1)/float(result.shape[1])) > minimum_detection_rate]
		print('We keep %d genes after discarding the lowly expressed genes (detected in less than %.2f%% of the ST spots)'%(result.shape[0],100.0*minimum_detection_rate))

	# print the median sequencing depth
	print('The median sequencing depth across the ST spots is %d'%(np.median(result.sum(0))))

	# write the modified count files back to the disk
	for filename in result.columns.levels[0]:
		result[filename].to_csv(filename+suffix,sep='\t',index=True)



if __name__ == '__main__':
	data_dir = '../data/BA44_testdata'
	spaceranger_dirs = sorted(glob.glob(os.path.join(data_dir, 'spaceranger', '*')))
	fullres_image_files = sorted(glob.glob(os.path.join(data_dir, 'fullres_images', '*.jpg')))
	annot_files = sorted(glob.glob(os.path.join(data_dir, 'annotations', '*.csv')))

	# Image-only datasets	
	gdat = create_visium_dataset(spaceranger_dirs, use_count=False, use_image=True, annot_files=annot_files,
		fullres_image_files=fullres_image_files, spatial=True)
	print(len(gdat))
	x, y = gdat[0]
	print(x.shape)
	print(y.min(), y.max())
	
	pdat = create_visium_dataset(spaceranger_dirs, use_count=False, use_image=True, annot_files=annot_files,
		fullres_image_files=fullres_image_files, spatial=False)
	print(len(pdat))
	x, y = pdat[0]
	print(x.shape)
	print(y)
	print(pdat.imgpath_mapping[0])

	# Count-only datasets
	pdat = create_visium_dataset(spaceranger_dirs, use_count=True, use_image=False, annot_files=annot_files,
		spatial=False)
	print(len(pdat))
	x,y = pdat[0]
	print(x, x.min(), x.max())
	print(y, y)

	gdat = create_visium_dataset(spaceranger_dirs, use_count=True, use_image=False, annot_files=annot_files,
		spatial=True)
	print(len(gdat))
	x,y = gdat[0]
	print(x.shape, x.min(), x.max())
	print(y, y.min(), y.max())

	#data_dir = '/Volumes/Aidan_NYGC/Visium/Human_SC_20230419/'
	#spaceranger_dirs = glob.glob(os.path.join(data_dir, 'spaceranger', '*'))
	#create_visium_dataset(spaceranger_dirs, use_count=True, minimum_detection_rate=0.02)
