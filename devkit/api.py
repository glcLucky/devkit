# -*- coding: utf-8 -*-

"""
api.py
@author: Jasper Gui
@email: jasper.gui@outlook.com
@date: 2020.08.15
"""

# data viz using dimension reudction method
from . src.utils.viz import mgc_scatter
from . src.utils.viz import tsne_reducer
from . src.utils.viz import umap_reducer
from . src.utils.viz import data_viz

from . src.utils.viz import plot_feature_importances

from .src.utils.viz import mgcMultiHeat
from .src.utils.viz import mgcVizMseDistribution

from .src.utils.viz import plot_cluster_representation

# confusion matrix
from . src.utils.model import make_confusion_matrix
from . src.utils.model import Hinton_mat

from . src.utils.model import clustering_accuracy


# mgc Multiclassed Classfication
from . src.mgcMulticlass import mgcplot