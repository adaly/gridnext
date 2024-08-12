import os
import scanpy as sc 
from gridnext.llm import scBERT
from gridnext.densenet import DenseNet
from gridnext.multimodal_datasets import MMAnnGridDataset
from gridnext.gridnet_models import GridNetHexMM

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# Data paths
data_dir = '/Volumes/Aidan_NYGC/Visium/Human_SC_GridNext/'
adata_path = os.path.join(data_dir, 'adata_human_spc_counts_images_scbertpp.h5ad')
densenet_path = os.path.join(data_dir, 'gridnext_models', 'f_img_20240322.pth')
scbert_path = os.path.join(data_dir, 'scbert_models', 'human_spc_vis_finetune_p0_best.pth')
corrector_path = os.path.join(data_dir, 'gridnext_models', 'glayer_mm_2024-05-22.pth')


# Helper function for loading model parameters trained on CUDA to CPU
def load_map(paramfile):
	if torch.cuda.is_available():
		data = torch.load(paramfile)
	else:
		data = torch.load(paramfile, map_location=torch.device('cpu'))
	return data

# Image transforms to be applied prior to input to DenseNet image classifier (f)
ppx = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
patch_size = (3,224,224)
H_VISIUM = 78
W_VISIUM = 64


# Load multimodal AnnData object
# (preprocessed for scBERT using gridnext.llm.preprocess_scbert)
adata = sc.read_h5ad(adata_path)
mm_data = MMAnnGridDataset(adata, obs_label='annotation', obs_arr='array', obs_img='imgpath', 
	img_transforms=ppx)

n_genes = adata.shape[1]
n_classes = len(mm_data.classes)


# Instantiate DenseNet model and load pretrained parameters
f_img = DenseNet(num_classes=n_classes, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
densenet_pdict = load_map(densenet_path)
f_img.load_state_dict(densenet_pdict)


# Instantiate scBERT model and load pretrained parameters
f_cnt = scBERT(n_classes=n_classes)
scbert_ckpt = load_map(scbert_path)
f_cnt.load_state_dict(scbert_ckpt['model_state_dict'])


# Instantiate multi-modal GridNext model and load parameters for corrector (g-network)
g_mm = GridNetHexMM(f_img, f_cnt, 
	image_shape=patch_size, count_shape=(n_genes,), grid_shape=(H_VISIUM, W_VISIUM), n_classes=n_classes)
corrector_pdict = load_map(corrector_path)
g_mm.corrector.load_state_dict(corrector_pdict)


# Test model
'''
dl = DataLoader(mm_data, batch_size=1)
x, y = next(iter(dl))

g_mm.eval()
with torch.no_grad():
	preds = g_mm(x)
	print(preds.shape)
'''
