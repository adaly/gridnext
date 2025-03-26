import torch
import torch.nn as nn

import numpy as np

# import the checkpoint API 
import torch.utils.checkpoint as cp

# Support for convolutions over hexagonally packed grids
import hexagdly


# Helper function for random weight initialization of model layers
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Base class for GridNet on Cartesian array data (e.g., non-Visium)
class GridNet(nn.Module):
    def __init__(self, patch_classifier, patch_shape, grid_shape, n_classes, 
        use_bn=True, atonce_patch_limit=None, f_dim=None):
        super(GridNet, self).__init__()

        self.patch_shape = patch_shape
        self.grid_shape = grid_shape
        self.n_classes = n_classes
        self.patch_classifier = patch_classifier
        self.use_bn = use_bn
        self.atonce_patch_limit = atonce_patch_limit
        
        # If output of patch classifier is different dimension from n_classes
        if f_dim is None:
            f_dim = n_classes
        self.f_dim = f_dim

        self.corrector = self._init_corrector()

        # NOTE: This tensor MUST have requires_grad=True for gradient checkpointing to execute. Otherwise, 
        # it fails during backprop with "element 0 of tensors does not require grad and does not have a grad_fn"
        self.bg = torch.zeros((1,f_dim), requires_grad=True)
        self.register_buffer("bg_const", self.bg) # Required for proper device handling with CUDA.

        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.register_buffer("dummy_tensor", self.dummy)

    # Define Sequential model containing convolutional layers in global corrector.
    def _init_corrector(self):
        cnn_layers = []
        cnn_layers.append(nn.Conv2d(self.f_dim, self.n_classes, 3, padding=1))
        if self.use_bn:
            cnn_layers.append(nn.BatchNorm2d(self.n_classes))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Conv2d(self.n_classes, self.n_classes, 5, padding=2))
        if self.use_bn:
            cnn_layers.append(nn.BatchNorm2d(self.n_classes))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Conv2d(self.n_classes, self.n_classes, 5, padding=2))
        if self.use_bn:
            cnn_layers.append(nn.BatchNorm2d(self.n_classes))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Conv2d(self.n_classes, self.n_classes, 3, padding=1))
        return nn.Sequential(*cnn_layers)

    # Wrapper function that calls patch classifier on foreground patches and returns constant values for background.
    def foreground_classifier(self, x):
        if torch.max(x) == 0:
            return self.bg_const
        else:
            return self.patch_classifier(x.unsqueeze(0))

    # Helper function to make checkpointing possible in patch_predictions
    def _ppl(self, patch_list, dummy_arg=None):
        assert dummy_arg is not None
        #return torch.cat([self.foreground_classifier(p) for p in patch_list], 0)
        return self.patch_classifier(patch_list)

    def patch_predictions(self, x):
        # Reshape input tensor to be of shape (batch_size * h_grid * w_grid, channels, h_patch, w_patch).
        patch_list = torch.reshape(x, (-1,)+self.patch_shape)

        if self.atonce_patch_limit is None:
            patch_pred_list = self._ppl(patch_list, self.dummy_tensor)
        # Process flattened patch list in fixed-sized chunks, checkpointing the result for each.
        else:
            cp_chunks = []
            count = 0
            while count < len(patch_list):              
                length = min(self.atonce_patch_limit, len(patch_list)-count)
                tmp = patch_list.narrow(0, count, length)

                # NOTE: Without at least one input to checkpoint having requires_grad=True, then the gradient tape
                # will break and gradients for patch classifier will be None/0.
                if any(p.requires_grad for _,p in self.patch_classifier.named_parameters()):
                    chunk = cp.checkpoint(self._ppl, tmp, self.dummy_tensor)
                else:
                    chunk = self._ppl(tmp, self.dummy_tensor)

                cp_chunks.append(chunk)
                count += self.atonce_patch_limit
            patch_pred_list = torch.cat(cp_chunks,0)

        patch_pred_grid = torch.reshape(patch_pred_list, (-1,)+self.grid_shape+(self.f_dim,))
        patch_pred_grid = patch_pred_grid.permute((0,3,1,2))

        return patch_pred_grid

    def forward(self, x):
        patch_pred_grid = self.patch_predictions(x)

        # Apply global corrector.
        corrected_grid = self.corrector(patch_pred_grid)
        
        return corrected_grid


# Extension of GridNet that performs convolution over hexagonally-packed grids.
# Expects input to employ the addressing scheme employed by HexagDLy.
class GridNetHex(GridNet):
    def __init__(self, patch_classifier, patch_shape, grid_shape, n_classes, 
        use_bn=True, atonce_patch_limit=None, f_dim=None):
        super(GridNetHex, self).__init__(patch_classifier, patch_shape, grid_shape, n_classes, 
            use_bn, atonce_patch_limit, f_dim)

    def _init_corrector(self):
        cnn_layers = []
        cnn_layers.append(hexagdly.Conv2d(in_channels=self.f_dim, out_channels=32, 
            kernel_size=1, stride=1, bias=True))
        cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
            kernel_size=1, stride=1, bias=True))
        if self.use_bn:
            cnn_layers.append(nn.BatchNorm2d(32))
        cnn_layers.append(nn.ReLU())

        cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
            kernel_size=1, stride=1, bias=True))
        cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
            kernel_size=1, stride=1, bias=True))
        if self.use_bn:
            cnn_layers.append(nn.BatchNorm2d(32))
        cnn_layers.append(nn.ReLU())

        cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=self.n_classes, 
            kernel_size=1, stride=1, bias=True))
        return nn.Sequential(*cnn_layers)


# Like GridNetHex, but:
# - If spot descriptor is 1D (e.g., count vector), inputs are (batch, feats, H_ST, W_ST).
#   If spot descriptor is >1D (e.g., RGB image), inputs are (batch, H_ST, W_ST, feats1, ... featsD).
# - Expects inputs in odd-right indexing scheme provided by Visium, handles rotation and flipping
#   to and from the odd-down indexing scheme of HexagDLy under the hood.
#
# NOTE: torch.flip (and torch.rot90?) returns a copy of the tensor, which is more time consuming,
# but the gradient tape seems to remain intact (optimizing all parameters updates patch_classifier).
class GridNetHexOddr(GridNetHex):
    
    # For both forward() and patch_predictions():
    #  1D: (batch, feats, H_ST, W_ST) -> (batch, n_class, H_ST, W_ST)
    # >1D: (batch, H_ST, W_ST, feats1, ..., featsD) -> (batch, n_class, H_ST, W_ST)
    
    def patch_predictions(self, x):
        # Reshape inputs with 1D spot features s.t. we can use original implementation.
        if len(x.shape) == 4:
            x_perm = x.permute((0,2,3,1))
            return super(GridNetHexOddr, self).patch_predictions(x_perm)

        return super(GridNetHexOddr, self).patch_predictions(x)
    
    def forward(self, x):
        # Patch prediction output: (batch, n_class, H_ST, W_ST)
        patch_pred_grid = self.patch_predictions(x)
        
        # Clockwise rotate & horizontal flip to odd-down indexing before HexagDLy convolution
        ppred_rot = torch.rot90(patch_pred_grid, 1, [3,2])
        ppred_rot = torch.flip(ppred_rot, [3])
                        
        corrected_grid = self.corrector(ppred_rot)
        
        # Horizontal flip & counter-clockwise rotate back to odd-right indexing to match Visium input
        out_rot = torch.flip(corrected_grid, [3])
        out_rot = torch.rot90(out_rot, 1, [2,3])
        
        return out_rot


# Contains two classifiers: image (operating on 3d inputs) and count (operating on 1d inputs)
# - Operates on tuples of (image, count) tensor inputs
# - After applying classifiers to each modalities, concatenates along feature dimension before applying corrector
class GridNetHexMM(GridNetHexOddr):
    def __init__(self, image_classifier, count_classifier, image_shape, count_shape, grid_shape, n_classes,
        use_bn=True, atonce_patch_limit=None, atonce_count_limit=None, device="cpu", delay_sending_to_device=True, image_f_dim=None, count_f_dim=None):
        if image_f_dim is None:
            image_f_dim = n_classes
        if count_f_dim is None:
            count_f_dim = n_classes

        super(GridNetHexMM, self).__init__(image_classifier, image_shape, grid_shape, n_classes,
            use_bn, atonce_patch_limit, image_f_dim+count_f_dim)

        self.image_classifier = image_classifier
        self.count_classifier = count_classifier
        self.image_shape = image_shape
        self.count_shape = count_shape
        self.image_f_dim = image_f_dim
        self.count_f_dim = count_f_dim

        self.mm_atonce_patch_limit=atonce_patch_limit
        #atone patch limit exists already in the super class
        self.mm_atonce_count_limit = atonce_count_limit
        self.delay_sending_to_device = delay_sending_to_device
        self.mode="count"        

    def _set_mode(self, mode):
        if mode == 'count':
            self.patch_classifier = self.count_classifier
            self.patch_shape = self.count_shape
            self.f_dim = self.count_f_dim
            self.mode="count"
            self.atonce_patch_limit=self.mm_atonce_count_limit
        elif mode == 'image':
            self.patch_classifier = self.image_classifier
            self.patch_shape = self.image_shape
            self.f_dim = self.image_f_dim
            self.mode="image"
            self.atonce_patch_limit = self.mm_atonce_patch_limit
        else:
            self.f_dim = self.count_f_dim + self.image_f_dim

    # Separately make patchwise predictions using each modality (and appropriate f-network), 
    #   then concat along feature dimension
    def patch_predictions(self, x):
        x_image, x_count = x

        self._set_mode('count')

        if self.delay_sending_to_device:
            x_image=x_image.to("cpu") #if the image was in the GPU at this point it may be that
            # it remains there and there is a copy in the cpu and we lose the reference to it
            x_count=x_count.to(self.device)
            torch.cuda.empty_cache()

        # LES: I will accelerate this testing temporarily:
        ppg_count = super(GridNetHexMM, self).patch_predictions(x_count)
        
        self._set_mode('image')
        if self.delay_sending_to_device:
            x_image = x_image.to(self.device)
            x_count = x_count.to("cpu")
            torch.cuda.empty_cache()

        ppg_image = super(GridNetHexMM, self).patch_predictions(x_image)
        
        del x_count,x_image
        torch.cuda.empty_cache()

        if self.delay_sending_to_device:
            ppg_image = ppg_image.to(self.device)
            ppg_count = ppg_count.to(self.device)

        self._set_mode('concat')
        
        return torch.cat((ppg_count, ppg_image), dim=1)
