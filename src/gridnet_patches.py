import torch
import torch.nn as nn

import numpy as np

# import the checkpoint API 
import torch.utils.checkpoint as cp

# Support for convolutions over hexagonally packed grids
import hexagdly

class GridNet(nn.Module):
	def __init__(self, patch_classifier, patch_shape, grid_shape, n_classes, 
		use_bn=True, atonce_patch_limit=None):
		super(GridNet, self).__init__()

		self.patch_shape = patch_shape
		self.grid_shape = grid_shape
		self.n_classes = n_classes
		self.patch_classifier = patch_classifier
		self.use_bn = use_bn
		self.atonce_patch_limit = atonce_patch_limit

		self.corrector = self._init_corrector()

		# Define a constant vector to be returned by attempted classification of "background" patches
		#self.bg = torch.zeros((1,n_classes))
		# NOTE: This tensor MUST have requires_grad=True for gradient checkpointing to execute. Otherwise, 
		# it fails during backprop with "element 0 of tensors does not require grad and does not have a grad_fn"
		self.bg = torch.zeros((1,n_classes), requires_grad=True)
		self.register_buffer("bg_const", self.bg) # Required for proper device handling with CUDA.

		self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
		self.register_buffer("dummy_tensor", self.dummy)

	# Define Sequential model containing convolutional layers in global corrector.
	def _init_corrector(self):
		cnn_layers = []
		cnn_layers.append(nn.Conv2d(self.n_classes, self.n_classes, 3, padding=1))
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

		patch_pred_grid = torch.reshape(patch_pred_list, (-1,)+self.grid_shape+(self.n_classes,))
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
		use_bn=True, atonce_patch_limit=None):
		super(GridNetHex, self).__init__(patch_classifier, patch_shape, grid_shape, n_classes, 
			use_bn, atonce_patch_limit)

	# Note: hexagdly.Conv2d seems to provide same-padding when stride=1.
	'''def _init_corrector(self):
		cnn_layers = []
		cnn_layers.append(hexagdly.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes, 
			kernel_size=1, stride=1, bias=True))
		if self.use_bn:
			cnn_layers.append(nn.BatchNorm2d(self.n_classes))
		cnn_layers.append(nn.ReLU())
		cnn_layers.append(hexagdly.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes, 
			kernel_size=1, stride=1, bias=True))
		return nn.Sequential(*cnn_layers)
	'''

	# More complex model to make sure there is sufficient complexity to memorize training data:
	# Conv2d(32)->Conv2d(32)->BN->ReLU->Conv2d(32)->Conv2d(32)->BN->ReLU->Conv2d(n_classes)
	def _init_corrector(self):
		cnn_layers = []
		cnn_layers.append(hexagdly.Conv2d(in_channels=self.n_classes, out_channels=32, 
			kernel_size=1, stride=1, bias=True))
		cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
			kernel_size=1, stride=1, bias=True))
		if self.use_bn:
			cnn_layers.append(nn.BatchNorm2d(self.n_classes))
		cnn_layers.append(nn.ReLU())

		cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
			kernel_size=1, stride=1, bias=True))
		cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=32, 
			kernel_size=1, stride=1, bias=True))
		if self.use_bn:
			cnn_layers.append(nn.BatchNorm2d(self.n_classes))
		cnn_layers.append(nn.ReLU())

		cnn_layers.append(hexagdly.Conv2d(in_channels=32, out_channels=self.n_classes, 
			kernel_size=1, stride=1, bias=True))
		return nn.Sequential(*cnn_layers)


def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		nn.init.zeros_(m.bias)
	if type(m) == nn.BatchNorm2d:
		nn.init.ones_(m.weight)
		nn.init.zeros_(m.bias)
