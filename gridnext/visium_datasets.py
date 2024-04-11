import os
import csv
import gzip
import glob
import logging
import scipy.io
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from gridnext.utils import visium_get_positions, visium_find_position_file
from gridnext.imgprocess import save_visium_patches, VISIUM_H_ST, VISIUM_W_ST, distance_um_to_px
from gridnext.image_datasets import PatchDataset, PatchGridDataset
from gridnext.count_datasets import CountDataset, CountGridDataset
from gridnext.multimodal_datasets import MMStackDataset


# Creates and returns an appropriate Dataset subclass for the modalities specified
def create_visium_dataset(spaceranger_dirs, use_count=True, use_image=True, spatial=True,
	annot_files=None, fullres_image_files=None, count_suffix=".unified.tsv.gz", minimum_detection_rate=0.02,
	patch_size_px=None, patch_size_um=100.0, img_transforms=None, select_genes=None, save_patches_to=None):
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
	patch_size_px: int or None
		width of patches, in pixels, to be extracted at each spot location. Supercedes patch_size_um
	patch_size_um: float or None
		width of patches, in um, to be extracted at each spot location. Resolution is inferred from Spaceranger position file.
	img_transforms: torchvision.transform
		transform to be applied to each image patch upon loading (e.g., normalization for pretrained network)
	select_genes: iterable of str
		list of genes to subset from the full transcriptome
	save_patches_to: path or None
		path to top-level directory in which to save image patches (one sub-directory created per array);
		or None to save in-place in Spaceranger directory for each array

	Returns:
	-------
	PatchDataset or PatchGridDataset or CountDataset or CountGridDataset or MultiModalDataset or
		MultiModalGridDataset
		appropriate subclass of torch.utils.data.Dataset for learning task
	'''
		
	if not (use_count or use_image):
		raise ValueError("Must utilize at least one data modality")
	if use_image and not (patch_size_px or patch_size_um):
		raise ValueError("Must specify patch size in pixels (int) or um (float)")

	# Check if unified countfiles have already been generated for these data
	if use_count:
		count_files = [os.path.join(srd, Path(srd).name+count_suffix) for srd in spaceranger_dirs]

		if not np.all([os.path.exists(cfile) for cfile in count_files]):
			print("No unified countfiles detected (%s) -- generating..." % ("*"+count_suffix))

			visium_prepare_count_files(spaceranger_dirs, count_suffix, minimum_detection_rate)

	# Check if image patches have already been extracted for these data
	if use_image:
		if patch_size_px is not None:
			patch_suffix = '_patches%dpx' % patch_size_px
		else:
			patch_suffix = '_patches%dum' % patch_size_um

		if save_patches_to is None:
			patch_dirs = [os.path.join(srd, Path(srd).name+patch_suffix) for srd in spaceranger_dirs]
		else:
			if not os.path.exists(save_patches_to):
				os.mkdir(save_patches_to)
			patch_dirs = [os.path.join(save_patches_to, Path(srd).name+patch_suffix) for srd in spaceranger_dirs]

		if not np.all([os.path.exists(pdir) for pdir in patch_dirs]):
			print("No extracted image patches detected (%s) -- generating..." % ("*"+patch_suffix))

			if fullres_image_files is None:
				raise ValueError('Must provide fullres_image_files to extract image patches')

			for imfile, pdir, srd in zip(fullres_image_files, patch_dirs, spaceranger_dirs):
				if not os.path.exists(imfile):
					raise ValueError('Could not find image file: %s' % imfile)

				if patch_size_px is None:
					ps = distance_um_to_px(srd, patch_size_um)
				else:
					ps = patch_size_px

				save_visium_patches(imfile, spaceranger_dir=srd, dest_dir=pdir, patch_size=ps)

	# Find position files mapping spot barcodes to array/pixel coordinates
	position_files = [visium_find_position_file(srd) for srd in spaceranger_dirs]

	if spatial:
		if use_image:
			dat_image = PatchGridDataset(patch_dirs, annot_files=annot_files, position_files=position_files,
				Visium=True, img_transforms=img_transforms, h_st=VISIUM_H_ST, w_st=VISIUM_W_ST)
		if use_count:
			dat_count = CountGridDataset(count_files, annot_files=annot_files, position_files=position_files,
				Visium=True, select_genes=select_genes, h_st=VISIUM_H_ST, w_st=VISIUM_W_ST)
	else:
		if use_image:
			dat_image = PatchDataset(patch_dirs, annot_files=annot_files, position_files=position_files,
				Visium=True, img_transforms=img_transforms)
		if use_count:
			dat_count = CountDataset(count_files, annot_files=annot_files, position_files=position_files,
				Visium=True, select_genes=select_genes)

	if use_image and use_count:
		if not spatial:
			raise NotImplementedError("Need to ensure indexing matches between spots in datasets")
		return MMStackDataset(dat_image, dat_count)
	elif use_image:
		return dat_image
	else:
		return dat_count


# Generate unified countfiles containing same genes in same order from a list of spaceranger directories
def visium_prepare_count_files(spaceranger_dirs, suffix, minimum_detection_rate=None):
	# Assemble count matrix dataframe from components in output directories:
	frames = []
	count_files = []
	for srd in spaceranger_dirs:
		df = read_feature_matrix(srd)

		positions = visium_get_positions(srd)
		positions_list = []
		for b in df.columns:
			xcoor = positions.loc[b,'array_col']
			ycoor = positions.loc[b,'array_row']
			positions_list.append('%d_%d' % (xcoor, ycoor))
		df.columns = positions_list

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

def read_feature_matrix(srd):
	matrix_dir = os.path.join(srd, "outs/filtered_feature_bc_matrix/")
	mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

	features_path = os.path.join(matrix_dir, "features.tsv.gz")
	feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, "rt"), delimiter="\t")]

	barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
	barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, "rt"), delimiter="\t")]

	df = pd.DataFrame.sparse.from_spmatrix(mat, index=feature_ids, columns=barcodes)
	return df 

# Create an AnnData object containing the annotated count data from multiple Visium arrays
def create_visium_anndata(spaceranger_dirs, annot_files=None, destfile=None):
	'''
	Parameters:
	----------
	spaceranger_dirs: iterable of path
		path to spaceranger output directories for each Visium array in dataset
	annot_files: iterable of path, or None
		path to Loupe annotation file for each array in dataset; None for un-annotated data
	destfile: path or None
		path in which to save generated AnnData object
	'''
	adata_list = []

	for i, srd in enumerate(spaceranger_dirs):
		df_counts = read_feature_matrix(srd).T
		df_pos = visium_get_positions(srd)

		barcodes = df_pos[df_pos['in_tissue']==1].index

		if annot_files is not None:
			df_annot = pd.read_csv(annot_files[i], header=0, index_col=0, sep=',')
			df_annot = df_annot.loc[df_annot.iloc[:,0] != '']  # filter un-annotated spots
			barcodes = barcodes.intersection(df_annot.index)

		arr = Path(srd).stem

		obs = pd.DataFrame({
			'x':df_pos.loc[barcodes, 'array_col'], 
			'y':df_pos.loc[barcodes, 'array_row'],
			'x_px':df_pos.loc[barcodes, 'pxl_col_in_fullres'],
			'y_px':df_pos.loc[barcodes, 'pxl_row_in_fullres'],
			'array': arr
		})
		if annot_files is not None:
			obs['annotation'] = df_annot.loc[barcodes].iloc[:,0]
		obs.index = ['%s_%d_%d' % (arr,x,y) for x,y in zip(obs['x'].values, obs['y'].values)]

		adata = ad.AnnData(X=sparse.csr_matrix(df_counts.loc[barcodes, :].values), 
			var=pd.DataFrame(index=df_counts.columns), 
			obs=obs)
		adata_list.append(adata)

	adata_all = ad.concat(adata_list, axis=0, join='outer')

	if destfile is not None:
		adata_all.write(destfile, compression='gzip')

	return adata_all

# Create an AnnData object containing (annotated) count and image data from multiple Visium arrays.
# Stores only path to extracted image file per spot.
def create_visium_anndata_img(spaceranger_dirs, imgpatch_dirs=None, fullres_image_files=None,
	annot_files=None, destfile=None, patch_size_px=None, patch_size_um=100.0, save_patches_to=None):
	'''
	Parameters:
	----------
	spaceranger_dirs: iterable of path
		path to spaceranger output directories for each Visium array in dataset
	imgpatch_dirs: iterable of path, or None
		path to directory containing extracted image patches for each spot (formatted [array]_[x]_[y].jpg)
	fullres_image_files: iterable of path, or None
		path to full-resolution image file for each Visium array (required if imgpatch_dirs is None)
	annot_files: iterable of path, or None
		path to Loupe annotation file for each array in dataset; None for un-annotated data
	destfile: path or None
		path in which to save generated AnnData object
	patch_size_px: int or None
		width of patches, in pixels, to be extracted at each spot location. Supercedes patch_size_um
	patch_size_um: float or None
		width of patches, in um, to be extracted at each spot location. Resolution is inferred from Spaceranger position file.
	save_patches_to: path or None
		path to top-level directory in which to save image patches (one sub-directory created per array);
		or None to save in-place in Spaceranger directory for each array
	'''
	adata_count = create_visium_anndata(spaceranger_dirs, annot_files=annot_files, destfile=None)

	if imgpatch_dirs is None and fullres_image_files is None:
		raise ValueError('Must provide either patched image directories or fullres images')

	elif imgpatch_dirs is None:
		if patch_size_px is not None:
			patch_suffix = '_patches%dpx' % patch_size_px
		else:
			patch_suffix = '_patches%dum' % patch_size_um

		if save_patches_to is None:
			imgpatch_dirs = [os.path.join(srd, Path(srd).name+patch_suffix) for srd in spaceranger_dirs]
		else:
			if not os.path.exists(save_patches_to):
				os.mkdir(save_patches_to)
			imgpatch_dirs = [os.path.join(save_patches_to, Path(srd).name+patch_suffix) for srd in spaceranger_dirs]

		# Extract image patches for all arrays from which they have not yet been
		# TODO: abstract from this and create_visium_dataset?
		for imfile, pdir, srd in zip(fullres_image_files, imgpatch_dirs, spaceranger_dirs):
			if not os.path.exists(pdir):
				if not os.path.exists(imfile):
					raise ValueError('Could not find image file: %s' % imfile)

				if patch_size_px is None:
					ps = distance_um_to_px(srd, patch_size_um)
				else:
					ps = patch_size_px
				save_visium_patches(imfile, spaceranger_dir=srd, dest_dir=pdir, patch_size=ps)

	# Subset Visium count AnnData to only contain patches for which there is image data available
	adata_list = []
	for srd, pdir in zip(spaceranger_dirs, imgpatch_dirs):
		arr = Path(srd).stem
		adata_arr = adata_count[adata_count.obs.array == arr]

		imfiles = [os.path.join(pdir, '%s_%d_%d.jpg' % (arr, x, y)) for x,y in zip(adata_arr.obs.x, adata_arr.obs.y)]
		adata_arr.obs['imgpath'] = imfiles

		# Keep only spots for which image data exist
		keep_inds = [os.path.exists(im) for im in imfiles]
		adata_arr = adata_arr[keep_inds]

		adata_list.append(adata_arr)

	adata_img = ad.concat(adata_list)

	if destfile is not None:
		adata_img.write(destfile, compression='gzip')

	return adata_img


if __name__ == '__main__':
	data_dir = '../data/BA44_testdata'
	spaceranger_dirs = sorted(glob.glob(os.path.join(data_dir, 'spaceranger', '*')))
	fullres_image_files = sorted(glob.glob(os.path.join(data_dir, 'fullres_images', '*.jpg')))
	annot_files = sorted(glob.glob(os.path.join(data_dir, 'annotations', '*.csv')))

	print(spaceranger_dirs)
	print(annot_files)

	#adat1 = create_visium_anndata(spaceranger_dirs, destfile='../data/adata_unannot.h5ad')
	adat2 = create_visium_anndata(spaceranger_dirs, annot_files=annot_files, destfile='../data/adata_annot.h5ad')

	'''
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
	'''
