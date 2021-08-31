import torch

# TODO: gridnet_patches migrated from st_gridnet; join modules together
from gridnet_patches import GridNetHex

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
