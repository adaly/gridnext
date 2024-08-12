import os
import glob
import json
import numpy as np
import pandas as pd

import pkgutil
from io import StringIO
from pathlib import Path

import torch
from torchvision import transforms

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from gridnext.utils import visium_get_positions

################### CONSTANTS ##################

VISIUM_H_ST = 78  # Visium arrays contain 78 rows (height)
VISIUM_W_ST = 64  # ...each row contains 64 spots (width)

############### HELPER FUNCTIONS ###############

def pseudo_hex_to_oddr(col, row):
	if row % 2 == 0:
		x = col/2
	else:
		x = (col-1)/2
	y = row
	return int(x), int(y)

def oddr_to_pseudo_hex(col, row):
	y_vis = row
	x_vis = col * 2
	if row % 2 == 1:
		x_vis += 1
	return  int(x_vis), int(y_vis)

def pseudo_hex_to_cartesian(c):
	x, y = c 
	x2 = x/2
	y2 = y*np.sqrt(3)/2
	return (x2,y2)

''' Functions from SpaCell for color cast removal (background -> white)
'''
def remove_color_cast(img):
	img = img.convert('RGB')
	img_array = np.array(img)

	# Calculate 99th percentile pixels values for each channel
	rp = np.percentile(img_array[:, :, 0].ravel(), q=99)
	gp = np.percentile(img_array[:, :, 1].ravel(), q=99)
	bp = np.percentile(img_array[:, :, 2].ravel(), q=99)
	
	# scale image based on percentile values
	return scale_rgb(img, 255 / rp, 255 / gp, 255 / bp)

def scale_rgb(img, r_scale, g_scale, b_scale):
	source = img.split()
	R, G, B = 0, 1, 2
	red = source[R].point(lambda i: i * r_scale)
	green = source[G].point(lambda i: i * g_scale)
	blue = source[B].point(lambda i: i * b_scale)
	return Image.merge('RGB', [red, green, blue])


''' Functions to gauge image resolution
'''
def pairwise_distances(points):
	'''
	Parameters:
	----------
	points: (n, 2) ndarray

	Returns:
	-------
	distances: (n-choose-2,) array
	'''
	distances = []
	for i in range(len(points)):
		for j in range(i+1, len(points)):
			d = np.sqrt(np.sum((points[i]-points[j])**2))
			distances.append(d)
	return np.array(distances) 

def distance_um_to_px(spaceranger_dir, distance_um):
	'''
	Returns:
	-------
	imgres: number of pixels spanning specified physical distance (in um)
	'''
	positions = visium_get_positions(spaceranger_dir)

	# only need to look at a subset of points to estimate resolution
	if len(positions) > 10:
		positions = positions.sample(n=10)

	arr_coords = zip(positions['array_col'], positions['array_row'])
	px_coords = np.array(list(zip(positions['pxl_col_in_fullres'], positions['pxl_row_in_fullres'])))
	cart_coords = np.array(list(map(pseudo_hex_to_cartesian, arr_coords)))

	px_dist = pairwise_distances(px_coords)
	cart_dist = pairwise_distances(cart_coords)

	d100 = np.mean(px_dist / cart_dist)
	return int(np.rint(distance_um * d100 / 100))

######## VISIUM ANNOTATION PROCESSING #######

# Convert Loupe annotation files (barcode, AAR pairs) to Splotch annotation files (AARs x spots matrix).
# Splotch does this conversion under the hood -- for now, GridNext expects Splotch formatted inputs.
def to_splotch_annots(loupe_annotations, spaceranger_dirs, dest_dir, include_annots=None):
	'''
	Parameters:
	----------
	loupe_annotations: iterable of path
		paths to Loupe annotation files.
	spaceranger_dirs: iterable of Path
		spaceranger directories corresponding to Loupe annotation files.
	dest_dir: path
		directory in which to save Splotch-formatted annotation files.
	include_annots: list of str or None
		ordered list of annotations to include in Splotch files.
		If None, all annotations found across Loupe files will be included in alphanumeric order.
	'''
	
	# If annotations not provided, extract list of all unique annotations across provided files.
	if include_annots is None:
		annot_list = []
		for afile in loupe_annotations:
			adat = pd.read_csv(afile, header=0, sep=',')
			keep_inds = [isinstance(a, str) and len(a) > 0 and a.lower() != 'undefined' for a in adat[adat.columns[1]]]
			annot_list.append(adat[adat.columns[1]][keep_inds])
		include_annots = list(np.unique(np.concatenate(annot_list)))

	for afile, srd in zip(loupe_annotations, spaceranger_dirs):
		annots = pd.read_csv(afile, header=0, sep=",")
		positions = visium_get_positions(srd)
		annot_matrix = np.zeros((len(include_annots), len(annots['Barcode'])), dtype=int)

		positions_list = []
		for i,b in enumerate(annots['Barcode']):
			xcoor = positions.loc[b,'array_col']
			ycoor = positions.loc[b,'array_row']
			positions_list.append('%d_%d' % (xcoor, ycoor))

			if annots.iloc[i,1] in include_annots:
				annot_matrix[include_annots.index(annots.iloc[i,1]),i] = 1

		splotch_frame = pd.DataFrame(annot_matrix, index=include_annots, columns=positions_list)
		outfile = os.path.join(dest_dir, Path(afile).name).replace('csv', 'tsv')
		splotch_frame.to_csv(outfile, sep='\t')


######## VISIUM IMAGE PROCESSING ########

# Extracts image patches centered at each Visium spot and returns as a 5D tensor for input to GridNet.
#   (tensor is odd-right indexed, meaning odd-indexed rows should be shifted right to generate hex grid.)
def grid_from_wsi_visium(fullres_imgfile, spaceranger_dir, patch_size=256, window_size=256, 
	preprocess_xform=None):
	'''
	Parameters:
	----------
	fullres_imgfile: path
		full-resolution image of tissue on Visium array.
	spaceranger_dir: path 
		directory containing output of Spaceranger
	patch_size: tuple of int
		size of image patches in pixels.
	window_size: tuple of int or float
		if different from patch_size, size of patches to be extracted before resizing to patch_size.
		If int, size of image region to be extracted in pixels. If float, fraction of patch_size.
	preprocess_transform: torchvision transform
		preprocessing transform to be applied to image patches extracted from WSI prior to inference.

	Returns:
	---------
	img_tensor: torch.Tensor
		tensor containing odd-right indexed representation of extracted image patches 
		(H_VISIUM, W_VISIUM, 3, patch_size, patch_size)
	'''
	img = np.array(Image.open(fullres_imgfile))
	ydim, xdim = img.shape[:2]

	if window_size is None:
		w = patch_size
	elif isinstance(window_size, float):
		w = int(window_size * xdim)
	elif isinstance(window_size, int):
		w = window_size
	else:
		raise ValueError("Window size must be a float or int")

	# Pad image such that no patches extend beyond image boundaries
	img = np.pad(img, pad_width=[(w//2, w//2), (w//2, w//2), (0,0)], mode='edge')

	df = visium_get_positions(spaceranger_dir)
	# Only consider spots that are within the tissue area.
	df = df[df['in_tissue']==1]

	# Create a 5D tensor to store the image array, then populate with patches
	# extracted from the full-resolution image.
	img_tensor = torch.zeros((VISIUM_H_ST, VISIUM_W_ST, 3, patch_size, patch_size))
	for i in range(len(df)):
		row = df.iloc[i]
		x_ind, y_ind = pseudo_hex_to_oddr(row['array_col'], row['array_row'])
		x_px, y_px = df.iloc[i]['pxl_col_in_fullres'], df.iloc[i]['pxl_row_in_fullres']
		x_px, y_px = int(np.rint(x_px)), int(np.rint(y_px))

		# Account for fractional pixel coordinates (rare)
		x_px = int(np.rint(x_px))
		y_px = int(np.rint(y_px))

		# Account for image padding
		x_px += w//2
		y_px += w//2

		patch = img[(y_px-w//2):(y_px+w//2), (x_px-w//2):(x_px+w//2)]
		patch = np.array(Image.fromarray(patch).resize((patch_size, patch_size)))
		
		patch = torch.from_numpy(patch).permute(2,0,1)
		if preprocess_xform is not None:
			xf = transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				preprocess_xform
			])
			patch = xf(patch)

		if y_ind >= VISIUM_H_ST or x_ind > VISIUM_W_ST:
			print("Warning: column %d row %d outside bounds of Visium array" % (x_ind, y_ind))
			continue

		img_tensor[y_ind, x_ind] = patch

	return img_tensor.float()

# For a sequence of samples, extracts patches centered around each Visium spot and save as JPG files 
#   in directory structure expected by Dataset classes for use with GridNet
def save_visium_patches(img_file, spaceranger_dir, dest_dir, patch_size=256, window_size=None):
	'''
	Parameters:
	-----------
	wsi_files: iterable of path
		paths to WSI files used in Visium pipeline.
	spaceranger_dirs: iterable of path
		output from Spaceranger associated with wsi_files.
	dest_dir: path
		top-level directory in which to save image patch data for all arrays (separate subdir will
		be created for each if it doesn't already exist).
	patch_size: tuple of int
		size of image patches in pixels.
	window_size: tuple of int or float
		if different from patch_size, size of patches to be extracted before resizing to patch_size.
		If int, size of image region to be extracted in pixels. If float, fraction of patch_size.
	'''

	# Generate patch grid tensor (H_ST, W_ST, C, H_p, W_p)
	patch_grid = grid_from_wsi_visium(img_file, spaceranger_dir, patch_size=patch_size, 
		window_size=window_size)

	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)

	slide = str(Path(spaceranger_dir).stem)

	# Save all foreground patches as separate JPG files.
	for oddr_x in range(VISIUM_W_ST):
		for oddr_y in range(VISIUM_H_ST):
			if patch_grid[oddr_y, oddr_x].max() > 0:
				patch = patch_grid[oddr_y, oddr_x]
				patch = np.moveaxis(patch.data.numpy().astype(np.uint8), 0, 2)  # switch to chanels-last

				# Save with Visium indexing to facilitate matching to count data
				x_vis, y_vis = oddr_to_pseudo_hex(oddr_x, oddr_y)
				Image.fromarray(patch).save(os.path.join(dest_dir, "%s_%d_%d.jpg" % (slide, x_vis, y_vis)), "JPEG")

# Multi-array analog of above; separate sub-directories created in dest_dir for each array.
def save_visium_patches_all(wsi_files, spaceranger_dirs, dest_dir, patch_size=256, window_size=None):
	if not os.path.isdir(dest_dir):
		os.mkdir(dest_dir)

	for img_file, srd in zip(wsi_files, spaceranger_dirs):
		print("%s : %s ..." % (img_file, srd))

		# Extract name of current tissue
		slide = str(Path(img_file).stem)
		dest_subdir = os.path.join(dest_dir, slide)
		save_visium_patches(img_file, srd, dest_subdir, patch_size, window_size)

# Create pseudo-Visium data (tissue_positions.csv and scalefactors_json.json) from a cropped image
def pseudo_visium_spots(fullres_roi, dest_dir, image_width_mm=8, spot_width_um=55, spot_spacing_um=100):
	'''
	Parameters:
	----------
	fullres_roi: path
		path to fullres image file that has been cropped to ROI that would fit on Visium array (e.g., 8mm^2)
	dest_dir: path
		directory in which to save pseudo-Visium files (tissue_positions.csv and scalefactors_json.json)
	image_width_um: float
		width of image ROI in mm
	spot_width_um: float
		width of simulated spots in um
	spot_spacing_um: float
		distance between adjacent spots in um
	'''
	resource_package = 'gridnext'
	pos_tpath = '/'.join(('visium_templates', 'tissue_positions.csv'))
	scale_tpath = '/'.join(('visium_templates', 'scalefactors_json.json'))

	pos_template = pkgutil.get_data(resource_package, pos_tpath).decode('utf-8')
	scale_template = pkgutil.get_data(resource_package, scale_tpath).decode('utf-8')

	df_pos = pd.read_csv(StringIO(pos_template), header=0, index_col=0, sep=',')
	dict_scale = json.loads(scale_template)

	img = np.array(Image.open(fullres_roi))
	w, h, c = img.shape 

	px_per_mm = w / image_width_mm
	spot_width_px = px_per_mm * spot_width_um / 1000
	spot_space_px = px_per_mm * spot_spacing_um / 1000

	ul = (int(np.rint(0.75*px_per_mm + spot_width_px/2)), 
		  int(np.rint(0.75*px_per_mm + spot_width_px/2)))
	dx = spot_space_px
	dy = spot_space_px * np.sqrt(3) / 2

	# Fill values in positions, scale factor templates
	for bc in df_pos.index:
		df_pos.loc[bc, 'pxl_col_in_fullres'] = int(np.rint(ul[0] + df_pos.loc[bc, 'array_col']/2 * dx))
		df_pos.loc[bc, 'pxl_row_in_fullres'] = int(np.rint(ul[1] + df_pos.loc[bc, 'array_row'] * dy))

	dict_scale['fiducial_diameter_fullres'] = dict_scale['fiducial_diameter_fullres'] / dict_scale['spot_diameter_fullres'] * spot_width_px
	dict_scale['spot_diameter_fullres'] = spot_width_px

	# Write files to '[array_name]/outs/spatial/tissue_positions.csv' and '[array_name]/outs/spatial/scalefactors_json.json'
	arr_name = Path(fullres_roi).stem.replace(' ', '_')
	pcomps = [dest_dir, arr_name, 'outs', 'spatial']
	out_path = ''

	for p in pcomps:
		out_path = os.path.join(out_path, p)
		if not os.path.exists(out_path):
			os.mkdir(out_path)
	df_pos.to_csv(os.path.join(out_path, 'tissue_positions.csv'))
	json.dump(dict_scale, open(os.path.join(out_path, 'scalefactors_json.json'), 'w+')) 
	


if __name__ == '__main__':
	#wsi_file = '/Users/adaly/Desktop/Visium/Maynard_ImageData/151510_full_image.tif'
	#tpl_file = '/Users/adaly/Documents/Splotch_projects/Maynard_DLPFC/data/Spaceranger_simulated/151510_tissue_positions_list.csv'
	#res = grid_from_wsi_visium(wsi_file, tpl_file)
	#print(res.max())

	data_dir = '/mnt/Shared01/BA44/'
	meta = pd.read_csv(os.path.join(data_dir, 'Splotch_Metadata.tsv'), header=0, sep='\t')

	wsi_files = [os.path.join(data_dir, imfile.replace('HE', 'HE_ccast')) for imfile in meta['Image file']]
	spaceranger_dirs = meta['Spaceranger output'].values

	dest_dir = os.path.join(data_dir, 'patchdata')
	save_visium_patches(wsi_files, spaceranger_dirs, dest_dir)

	#annot_files = [os.path.join(data_dir, afile) for afile in meta['Annotation file']]
	#to_splotch_annots(annot_files, tpl_files, os.path.join(data_dir, 'annotations_splotch'))

