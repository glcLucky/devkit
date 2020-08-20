import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import seaborn as sns


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

def clustering_accuracy(gtlabels, labels):
    """
    calculate the accuracy of clustering results
    """
    from scipy.optimize import linear_sum_assignment
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
        cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)

    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)

class Hinton_mat():
    def __init__(self):
        pass

    def fto_texts(self, ls_num):
        """
        round to 2 decimal for float and zero decimal for integer
        """
        if pd.Series(ls_num).dtypes == np.int32:
            _textx = "{0:.0f}".format(ls_num)
            _textz = {_textx == '0':''}.get(True, _textx)
        elif pd.Series(ls_num).dtypes == np.int64:
            _textx = "{0:.0f}".format(ls_num)
            textz = {_textx == '0':''}.get(True, _textx)
        elif pd.Series(ls_num).dtypes == np.float64:
            _textx = "{0:.2f}".format(ls_num)
            textz = {_textx == '0.00':''}.get(True, _textx)        
        return textz

    def _add_centered_square(self, ax, xy, area, lablz, text_size, **kwargs):
        """
        create hinton diagram element square with variable size according to weight matrix element value
        """
        size = np.sqrt(area)

        textz = self.fto_texts(lablz)
        loc = np.asarray(xy) - size/2.

        rect = mpatches.Rectangle(loc, size, size, **kwargs)

        label = ax.annotate(textz, xy=loc + size/2., fontsize=text_size, ha='center', va='center')

        ax.add_patch(rect)

    def _ftext_size(self, area, max_weight, plot_metric = None):
        """
        custom text size accroding to weight matrix element value
        """
        
        plot_metric = plot_metric if plot_metric is not None else "precision"
        
        min_thresh = max_weight/28 # previous 6

        _text_size = {area > 0 and area < min_thresh: 8, 
                    area >= min_thresh and area < 2*min_thresh: 10,
                    area >= 2*min_thresh: 14}.get(True, 0)

        return _text_size

    def _cm_metrics(self, cm, plot_metric = None):
        """
        convert basic confusion matrix to precsion or recall
        """
        plot_metric = plot_metric if plot_metric is not None else "precision"
        
        if plot_metric == 'bycount':
            cnf_matrix = cm
        
        elif plot_metric == 'recall':
            cnf_matrix = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(np.double).eps)           

        elif plot_metric == 'precision':
            cnf_matrix = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + np.finfo(np.double).eps)
        return cnf_matrix

    def _cm_color(self, val, max_weight, plot_metric=None):
        """
        weight matrix element color variant dependent on confusion matrix plot option - bycount, precision or recall
        """
        plot_metric = plot_metric if plot_metric is not None else "precision"

        min_thresh = max_weight/6
        
        color = {val > 0 and val < min_thresh: (1, 0, 0, 0.5),
                val >= min_thresh and val < 2*min_thresh: (1, 1, 0, 0.5),
                val >= 2*min_thresh : (0, 1, 0, 0.5)}.get(True, (1, 1, 1, 0.0))

        return color


    def hinton_mat(self, **kwargs):
        """
        return confusion matrix
        Arguments:
            title_plot: main title in plot
            max_weight: maximum size(in term of area) of the square based on sample size
            matrix: confusion matrix in numpy array
            plot_metric: by count, precision or recall
            dict_label: dictionary that maps integer to string class
            show_lgnd: boolean to indicate plot legend to be shown
            ax: plot axis object to be used, if None a new plot object will be created
            
        """
        
        def_vals = {"title_plot" : 'Top 1', 
                    "max_weight": None, 
                    "matrix" : None,
                    "plot_metric" : 'bycount',
                    "dict_label" : None, 
                    "show_lgnd": False, 
                    "ax": None,
                    # "figsizex": (10, 10)
                } # default parameters value

        for k, v in def_vals.items():
            kwargs.setdefault(k, v) # update keywords arguments if not provided by user
            
        title_plot = kwargs['title_plot']
        max_weight = kwargs['max_weight']
        matrix = kwargs['matrix']
        plot_metric = kwargs['plot_metric']
        dict_label = kwargs['dict_label']
        show_lgnd = kwargs['show_lgnd']
        ax = kwargs['ax']
        # figsizex = kwargs['figsizex']


        if dict_label is None:
            dict_label = {
                k: "label_{}".format(str(k).zfill(2)) 
                for k in range(len(matrix))}
   
        # Plot confusion metrics with Hinton method

        """
        Draw Hinton diagram for visualizing a weight matrix.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = figsizex)
        ax = ax if ax is not None else plt.gca()
        
        plot_metric = plot_metric if plot_metric is not None else "precision"
        
        dict_label = dict_label if dict_label is not None else dict(zip(range(matrix.shape[0]+1),range(matrix.shape[0]+1)))
        
        label_uniq = pd.Series(range(len(dict_label))).map(dict_label)
        
        ax.patch.set_facecolor('white')
        ax.set_aspect('equal', 'box')

        ax.set_yticks(np.arange(len(dict_label)))
        ax.set_yticklabels(label_uniq)
        
        ax.set_xticks(np.arange(len(dict_label)))
        ax.set_xticklabels(label_uniq, rotation=90)
        
        ax.grid(linestyle = 'dashed', color = 'lightgrey', alpha = 0.5)
        ax.patch.set_facecolor('None')
        ax.patch.set_alpha(0.0)
        matrix =self._cm_metrics(matrix, plot_metric = plot_metric)
        
        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
        
        # for weight matrix element    
        for xy, val in np.ndenumerate(matrix):

            color = self._cm_color(val, max_weight, plot_metric = plot_metric)
            text_size = self._ftext_size(val, max_weight, plot_metric = plot_metric)

            areaz = max(0.3, np.abs(val)/max_weight)

            self._add_centered_square(ax, np.asarray(xy), areaz, np.abs(val), text_size = text_size, color=color)

        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title("Confusion Matrix - {0}".format(title_plot), color='red')
    #     plt.show()

        if show_lgnd:      
            min_thresh = round(max_weight/8, 2)
            if plot_metric =="bycount":
                legend_lbl = [int(min_thresh*0.8), int(round(2*min_thresh)), int(round(5*min_thresh))]
            else:
                legend_lbl = [round(min_thresh*0.8,2), round(2*min_thresh,2), round(5*min_thresh,2)]

            msizes = [70, 200, 400]

            dict_msizes = dict(zip(msizes, legend_lbl))

            markers = []
            for size in dict_msizes:
                markers.append(plt.scatter([],[], s=size, label=dict_msizes[size], marker='s', facecolors='none', edgecolors = 'k'))
    
            plt.legend(handles = markers,
                    scatterpoints=1,
                    loc='best',
                    ncol=1, 
                    markerscale=1.6, 
                    numpoints=1,
                    borderpad=2, 
                    handlelength=3, 
                    labelspacing=2,
                    fontsize=14, frameon=False, 
                    bbox_to_anchor=(1,1))

    def plot_confusion_matrix(self, y_true, y_pred, dict_label=None, ncols=1, hgtunit=8, width=15):

        from sklearn.metrics import confusion_matrix

        conf_mat = confusion_matrix(y_true, y_pred)
        
        _plot_metric = ['bycount', 'precision', 'recall']
        
        sum_stats=True
        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(conf_mat) / float(np.sum(conf_mat))
        
            #if it is a binary confusion matrix, show some more stats
            if len(conf_mat)==2:
                #Metrics for Binary Confusion Matrices
                precision = conf_mat[1,1] / sum(conf_mat[:,1])
                recall    = conf_mat[1,1] / sum(conf_mat[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""
        
        nrows = int(np.ceil(len(_plot_metric) / ncols))
        hgt = int(nrows * hgtunit)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, hgt), sharex = False, sharey = False)
        
        for (axs_k, plot_m) in zip(axs.flatten(), ['bycount', 'precision', 'recall']):

            params = {"title_plot": plot_m + stats_text,
                      "max_weight": None,
                      "matrix": conf_mat,
                      "plot_metric": plot_m,
                      "dict_label": dict_label,
                      "show_lgnd": False,
                      'figsizex': (6, 6),
                      'ax': axs_k}
            self.hinton_mat(**params)
        fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
        
        fig.show()
