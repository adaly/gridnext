import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

from gridnext.utils import pseudo_to_true_hex, oddr_to_pseudo_hex


############### ROC and Precision-Recall Curves ###############

def performance_curves(true, smax, class_names=None, condition_names=None):
    '''
    Parameters:
    ----------
    true: (n_samples,) ndarray
        integer array containing class assignments for each sample.
    smax: (n_samples, n_classes) ndarray or list
        float array containing class probability vectors for each sample (columns sum to 1), 
        or list thereof (for multiple plots on the same axes)
    class_names: list of str, or None
        list of class names for sub-plots.
    condition_names: list of str or None
        names of conditions to be plotted on the same axis.
        
    Returns:
    ----------
    fig, ax: 
        figure and axis handles for a (2, n_class) plot array showing one-vs-rest ROC and 
        precision-recall curves for each class.
    macro_auroc: float
        macro average AUROC
    macro_auprc: float
        macro average AUPRC
    '''
    if isinstance(smax, list):
        n_classes = smax[0].shape[1]
        assert condition_names is not None, "Must provide names for each condition plotted"
    else:
        n_classes = smax.shape[1]
        smax = [smax]
        condition_names = ['']
    true_onehot = label_binarize(true, classes=list(range(n_classes)))
    
    n_col = 4
    n_row = int(np.ceil(n_classes/n_col)) * 2
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row),
                          constrained_layout=True)
    for i in range(n_row):
        for j in range(n_col):
            ax[i,j].axis('off')
    
    macro_auroc, macro_auprc = np.zeros((n_classes, len(smax))), np.zeros((n_classes, len(smax)))

    for c in range(n_classes):
        top_row = c // n_col
        btm_row = top_row + n_row // 2
        col = c % n_col
        
        ax[top_row,col].axis('on')
        ax[btm_row,col].axis('on')
        
        for i,s in enumerate(smax):
            fpr, tpr, _ = roc_curve(true_onehot[:,c], s[:,c])
            auroc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(true_onehot[:,c], s[:,c])
            auprc = auc(recall, precision)

            macro_auroc[c,i] = auroc
            macro_auprc[c,i] = auprc
        
            # Plot AUROC
            ax[top_row,col].plot(fpr, tpr, label='%s (AUC=%.3f)' % (condition_names[i], auroc))

            # Plot AUPRC
            ax[btm_row,col].plot(recall, precision, label='%s (AUC=%.3f)' % (condition_names[i], auprc))            
        
        ax[top_row,col].set_xlabel('FPR', fontsize=12)
        ax[top_row,col].set_ylabel('TPR', fontsize=12)
        
        ax[btm_row,col].set_xlabel('Recall', fontsize=12)
        ax[btm_row,col].set_ylabel('Precision', fontsize=12)
        
        ax[top_row,col].legend(fontsize=12)
        ax[btm_row,col].legend(fontsize=12)
        
        if class_names is not None:
            ax[top_row,col].set_title(class_names[c], fontsize=14)
            ax[btm_row,col].set_title(class_names[c], fontsize=14)
        ax[top_row,col].set_xlim(0,1)
        ax[top_row,col].set_ylim(0,1)
        ax[btm_row,col].set_xlim(0,1)
        ax[btm_row,col].set_ylim(0,1)
    
    return fig, ax, np.array(macro_auroc).mean(axis=0), np.array(macro_auprc).mean(axis=0)


############### Confusion Matrix Plots #################

def plot_confusion(y_true, y_pred, class_names=None, figsize=None):
    '''
    Parameters:
    ----------
    y_true: 1d array
        flattened array containing true label for each spot
    y_pred: 1d array
        flattened array containing predicted label for each spot
    class_names: iterable of str or None
        n_class-length iterable mapping foreground labels [1...n_class] to display names
    figsize: tuple
        figure size

    Returns:
    -------
    fig, ax: Figure, Axes objects
    '''
    cmat = confusion_matrix(y_true, y_pred)
    cmat_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    if class_names is None:
        class_names = np.unique(y_true)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    sns.heatmap(cmat_norm, annot=cmat, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, cbar_kws={'label':'fraction of spots'}
               )
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    return fig, ax

############### Misclassification Density Plots ###############

def misclass_density(out_softmax, true):    
    ydim, xdim = true.shape
    
    mcd = np.zeros((ydim, xdim))
    
    for y in range(ydim):
        for x in range(xdim):
            # Only care about foreground patches
            if true[y,x] > 0:
                p_correct = out_softmax[true[y,x]-1, y, x]
                mcd[y][x] = 1-p_correct
    return mcd

def plot_class_boundaries(base_image, true):
    ydim, xdim = true.shape
    
    fig, ax = plt.subplots(1)
    plt.axis("off")
    
    # Mask out background spots and render over black background
    masked_image = np.ma.masked_where(true==0, base_image)
    bgd = ax.imshow(np.zeros_like(true), cmap="gray")
    fgd = ax.imshow(masked_image, cmap="plasma")
    
    xpix = 1.0/xdim
    ypix = 1.0/ydim
        
    for y in range(ydim):
        for x in range(xdim):
            for x_off in [-1, 1]:
                if x+x_off < 0 or x+x_off >= xdim:
                    continue
                if true[y,x] != true[y,x+x_off]:
                    ax.axvline(x=x+x_off/2, ymin=1-((y+1)*ypix), ymax=1-(y*ypix), c='w')
            for y_off in [-1, 1]:
                if y+y_off < 0 or y+y_off >= ydim:
                    continue
                if true[y,x] != true[y+y_off,x]:
                    ax.axhline(y=y+y_off/2, xmin=x*xpix, xmax=(x+1)*xpix, c='w')
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(fgd, cax=cax)
    cbar.set_label("Misclassification Probability")
    
    return fig
    
############### Tissue Labeling Visualization ###############

def plot_label_tensor(label_tensor, class_names=None, Visium=False, ax=None, legend=True):
    '''
    Parameters:
    ----------
    label_tensor: Tensor
        2d Tensor object containing integer labeling of ST array
    class_names: iterable of str or None
        n_class-length iterable mapping foreground labels [1...n_class] to display names
    Visium: bool
        whether data are Visium formatted (i.e., implcitly hex-packed)
    ax: Axes or None
        existing Axes on which to plot labeling, or None to create new Axes
    legend: bool
        whether to draw a legend

    Returns:
    -------
    ax: Axes object
    '''
    if class_names is None:
        fg_vals = np.sort(np.unique(label_tensor[label_tensor > 0]))
    else:
        fg_vals = np.arange(1, len(class_names)+1)
                
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10,8))
    ax.set_aspect('equal')
    ax.invert_yaxis()
        
    for fgv in fg_vals:
        # transpose so first dimension is x (columns)
        pts = torch.nonzero(label_tensor.T==fgv)
            
        if class_names is None:
            lbl = fgv
        else:
            lbl = class_names[fgv-1]
        
        # convert to Cartesian coordinates if needed
        if Visium:
            pts = torch.tensor([pseudo_to_true_hex(*oddr_to_pseudo_hex(*c)) for c in pts])
            
        if len(pts) > 0:
            ax.scatter(pts[:,0], pts[:,1], label=lbl, s=10)
        else:
            ax.scatter([],[], label=lbl, s=10)
    
    ax.axis('off')
    if legend:
        ax.legend(bbox_to_anchor=(1,0), loc='lower left')
    
    return ax