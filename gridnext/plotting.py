import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc


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

# Accepts paired list of size (nsamples,) each containing 
def plot_confusion_matrix(y_true, y_pred, class_names, density):
    if np.min(y_true)>1:
        y_true -= 1
    if np.min(y_pred)>1:
        y_pred -= 1

    labels = range(0,len(class_names))
    cm_array = confusion_matrix(y_true,y_pred,labels=labels)
    
    fig, ax = plt.subplots(1, constrained_layout=True)
    if not density:
        cb = ax.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion matrix', fontsize=7)
        cbar = plt.colorbar(cb,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Number of spots', rotation=270, labelpad=30, fontsize=7)
    else:
        denom = cm_array.sum(1,keepdims=True)
        denom = np.maximum(denom, np.ones_like(denom))
        cb = ax.imshow(cm_array/denom.astype(float),
            vmin=0,vmax=1,interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Normalized confusion matrix', fontsize=7)
        cbar = plt.colorbar(cb,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Proportion of spots', rotation=270, labelpad=30, fontsize=7)

    xtick_marks = labels
    ytick_marks = labels
    ax.set_xticks(xtick_marks)
    ax.set_yticks(ytick_marks)
    ax.set_xticklabels(np.array(class_names),rotation=60,fontsize=7)
    ax.set_yticklabels(np.array(class_names),fontsize=7)
    ax.set_xlim([-0.5,len(class_names)-0.5])
    ax.set_ylim([len(class_names)-0.5,-0.5])
    ax.set_ylabel('True label',fontsize=7)
    ax.set_xlabel('Predicted label',fontsize=7)

    return fig

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
    