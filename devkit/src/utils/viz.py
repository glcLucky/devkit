import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def mgc_scatter(x, y, colors=None, includ_colorbar=False, title=None, figsize=(10, 10)):
    """
    plot scatter based on matplotlib
    @x <1d np.array>: data x
    @y <1d np.array>: data y
    @title <None or dict>: {'label': None, 'fontsize': 24}
    @colors <1d np.array>: data color, discrete values
    """
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=colors, cmap='Spectral', s=5)

    if colors is None:
        includ_colorbar = False

    if includ_colorbar:
        n_colors = len(set(colors)) + 1
        plt.colorbar(boundaries=np.arange(n_colors)-0.5).set_ticks(np.arange(n_colors))
    if title is not None:
        assert isinstance(title, dict)
        plt.title(**title)
    
def umap_reducer(X, n_components=2, **kwargs):
    """
    use umap to reduce the dimensionality of data
    @X <2d np.array>: the data to reduce
    @n_components <int>: the number of dim of reduced data
    return: the reducued data
    """
    from umap import UMAP
    assert len(X.shape) == 2
    umap_reducer = UMAP(n_components=2, **kwargs)
    umap_embedding = umap_reducer.fit_transform(X)
    return umap_embedding


def tsne_reducer(X, n_components=2, **kwargs):
    """
    use tsne to reduce the dimensionality of data
    @X <2d np.array>: the data to reduce
    @n_components <int>: the number of dim of reduced data
    return: the reducued data
    """
    from sklearn.manifold import TSNE
    assert len(X.shape) == 2
    tsne_reducer = TSNE(
        n_components=n_components, **kwargs)
    tsne_embedding = tsne_reducer.fit_transform(X)
    return tsne_embedding


def data_viz(X, y=None, method='umap', **kwargs):
    """
    viz high-dim data based on given dim reduction method
    @X <2d np.array>: the data to reduce
    @y <Nonr or 1d np.array>: the label of each sample
    @method <str>: the dim reduction method, available choices: ['umap', 'tsne']
    """
    if method == 'umap':
        reducer = umap_reducer
    elif method == 'tsne':
        reducer = tsne_reducer
    else:
        raise ValueError("Unrecognized dim reduction method: {}".format(method))
    embedding_2d = reducer(X, **kwargs)
    mgc_scatter(embedding_2d[:, 0],
            embedding_2d[:, 1],
            colors=y,
            includ_colorbar=True,
            figsize=(15, 10),
            title={'label': '{} projection of the embedding dataset'.format(method), 'fontsize': 24})


def plot_feature_importances(feat_imp, feat_labels, nlargest=10, fontsize=20, color_map="RdYlGn"):
    """
    bar plot of feature importances based on provided feat_imp array
    @feat_imp <1d np.array>: the feature importance array
    @feat_labels <1d np.array>: the corresponding feature names
    @nlargest <int>: only show the largest nlargest features
    """
    category_colors = plt.get_cmap(color_map)(np.linspace(0.15, 0.85, 10))
    ts_feat_imp = pd.Series(feat_imp, index=feat_labels)
    ts_feat_imp.nlargest(nlargest)[::-1].plot(kind='barh', figsize=(10,10), color=category_colors[::-1])
    plt.title("feature importance of the most {} importanct features".format(nlargest), fontsize=fontsize)


def mgcMultiHeat(**kwargs):
    """
    return multiple heatmap plot
    @X <np.array>: [n_samples, height, width, <channel, 3 or 4>]
    @arr_mse <1d np.array>: 
    """

    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np

    def_kwargs = {
        'X': None,
        'arr_mse': None,
        'arr_ids': None, # the ID of each image belonging to
        'arr_labels':None,  # the label of each image belonging to
        'arr_prob1': None,
        'category': None,
        'set_random': False,
        'figsize': (20, 10),
        'title_size': 5,
        'nrows': 7,
        'ncols': 5,
    }

    for k,v in def_kwargs.items():
        kwargs.setdefault(k, v)

    X = kwargs['X']
    arr_mse = kwargs['arr_mse']
    arr_ids = kwargs['arr_ids']
    arr_prob1 = kwargs['arr_prob1']
    arr_labels = kwargs['arr_labels']
    category = kwargs['category']
    set_random= kwargs['set_random']
    figsize = kwargs['figsize']
    title_size = kwargs['title_size']
    nrows = kwargs['nrows']
    ncols = kwargs['ncols']

    n_samples = len(X)
    n_images = nrows*ncols
    if set_random:
        if n_images < n_samples:
            idx_selected = np.random.choice(n_samples, nrows*ncols, replace=False)
        else:
            idx_selected = np.array(list(range(n_samples)))
    else:
        idx_selected = np.array(list(range(n_images)))
    
    X_sub = X[idx_selected]

    if arr_mse is not None:
        arr_mse_sub = arr_mse[idx_selected]

    if arr_ids is not None:
        arr_ids_sub = arr_ids[idx_selected]
    
    if arr_labels is not None:
        arr_labels_sub = arr_labels[idx_selected]

    print("total samples: {} / {}".format(len(X_sub), len(X)))
    
    if arr_ids is not None:
        print("unique samples: {} / {}".format(
            len(np.unique(arr_ids_sub)),
            len(np.unique(arr_ids))))
    

    fig = plt.figure(figsize=figsize)

    gird = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.45,
        share_all=True,
        cbar_location='right',
        cbar_mode='signle',
        cbar_size='7%',
        cbar_pad=0.15,        
        )

    for i, ax in enumerate(gird):
        im = ax.imshow(
            X_sub[i], 
            interpolation='spline16',
            cmap=plt.get_cmap('RdYlGn_r'))
        im.set_clim(vmin=0, vmax=1)
        title_ = ''

        if arr_ids is not None:
            title_ =  "id: {0}".format(arr_ids_sub[i])

        if arr_labels is not None:
            if title_ == '':
                title_ =  "label: {0}".format(arr_labels_sub[i])
            else:
                title_ +=  ", label: {0}".format(arr_labels_sub[i])

        if arr_mse is not None:
            if title_ == '':
                title_ =  "MSE {0:.3f}".format(arr_mse_sub[i])   
            else:
                title_ +=  ", MSE {0:.3f}".format(arr_mse_sub[i])

        if arr_prob1 is not None:
            if title_ == '':
                title_ =  "PROB1 {0:.3f}".format(arr_prob1[i])   
            else:
                title_ +=  ", PROB1 {0:.3f}".format(arr_prob1[i])
        if title_ != '':    
            ax.set_title(title_, size=title_size)
    
    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    
    if category is not None:
        fig.suptitle("Class ID {0}".format(category), fontsize=20, fontweight=0, color='black', style='italic', y=1.02)

    plt.show()  


def mgcVizMseDistribution(mse, outlier_ts=None, inlier_ts=None, figsize=(10,10), title_size=10):
    """
    viz the distribution of reconstruction error(or like)
    @mse <1d np.array>: the data to viz
    @outlier_ts <float>: the threshold value of outliers
    @inlier_ts <float>: the threshold value of inliers
    """



    def plt_hist(mse, ax=None, title_size=10):

        ts_mse = pd.Series(mse.flatten())
        if ax == None:
            fig = plt.figure(figsize=(14,7))
            ax = fig.add_subplot(111)

        ax.hist(pd.Series(mse.flatten()), alpha=0.9, histtype='step', fill=False, stacked=False, bins=20, log=False)
        ax.set_title("Deviation Histogram", fontsize = title_size)
        ax.set_xlabel("deviation degree", color = 'red', fontsize=24)
        ax.set_ylabel("frequency", color = 'gray', fontsize=24)
        ax.tick_params(axis='x', colors='black', labelsize=24)
        ax.tick_params(axis='y', colors='red', labelsize=24)
        ax.grid(True)

    def plot_scatter(mse, ax=None, title_size=10):
        if ax == None:
            fig = plt.figure(figsize=(14,7))
            ax = fig.add_subplot(111)
        ax.plot(list(range(len(mse))), mse, marker='o', ms=3.5, linestyle='',alpha = 0.05)
        if outlier_ts is not None:
            ax.hlines(outlier_ts, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='outlier_ts')
        if inlier_ts is not None:
            ax.hlines(inlier_ts, ax.get_xlim()[0], ax.get_xlim()[1], colors="g", zorder=100, label='inlier_ts')
        if outlier_ts or inlier_ts:
            ax.legend()
        ax.set_ylabel("reconstruction error",  color = 'gray', fontsize=24)
        ax.set_xlabel("data point index",  color = 'black', fontsize=24)
        ax.set_title("Reconstruction Error Scatter", fontsize = title_size)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    plot_scatter(mse, axs[0])
    plt_hist(mse, axs[1])

    plt.tight_layout()
    plt.show()
