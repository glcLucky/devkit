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
        n_colors = len(set(colors))
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
