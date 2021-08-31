import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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
    