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
    umap_reducer = UMAP(n_components=n_components, **kwargs)
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
    mgc_scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        colors=y,
        includ_colorbar=True,
        figsize=(15, 10),
        title={
            'label': '{} projection of the embedding dataset'.format(
                method), 'fontsize': 24})


def tsne_3d(**kwargs):
    
    """
    return scatter plot based on clustering
    
    Arguments:
        list of key performance indicators associated with each prediction class
    """
    from pandas import read_csv
    import colorlover as cl
    from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
    init_notebook_mode()
    import plotly.graph_objs as go
    import plotly

    # use default keyword parameters if user does not provide
    def_vals = {'ar_plot' : None,
                'dic_labels': None,
                'ar_marker_size': None,
                ''
                'file_dir' : "", 
                'plot_title': "Dimensionality Reduction <br>\ Latent Space"}

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    ar_plot = kwargs['ar_plot']
    dic_labels = kwargs['dic_labels']
    file_dir = kwargs['file_dir']
    plot_title = kwargs['plot_title']
    ar_marker_size = kwargs['ar_marker_size']
    
    if ar_marker_size is None:
        ar_marker_size = 2 
    def_color = ['rgb(31, 119, 180)',
                 'rgb(255, 127, 14)',
                 'rgb(44, 160, 44)',
                 'rgb(214, 39, 40)',
                 'rgb(148, 103, 189)',
                 'rgb(140, 86, 75)',
                 'rgb(227, 119, 194)',
                 'rgb(127, 127, 127)',
                 'rgb(188, 189, 34)',
                 'rgb(23, 190, 207)']

    def_color = ["rgb(230, 25, 75)", 
                 "rgb(60, 180, 75)", 
                 "rgb(255, 225, 25)", 
                 "rgb(0, 130, 200)", 
                 "rgb(245, 130, 48)", 
                 "rgb(145, 30, 180)", 
                 "rgb(70, 240, 240)", 
                 "rgb(240, 50, 230)", 
                 "rgb(210, 245, 60)", 
                 "rgb(250, 190, 190)", 
                 "rgb(0, 128, 128)", 
                 "rgb(230, 190, 255)", 
                 "rgb(170, 110, 40)", 
                 "rgb(255, 250, 200)", 
                 "rgb(128, 0, 0)", 
                 "rgb(170, 255, 195)", 
                 "rgb(128, 128, 0)", 
                 "rgb(255, 215, 180)", 
                 "rgb(0, 0, 128)", 
                 "rgb(128, 128, 128)"]

    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">    docomo ai</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.layout.Annotation(
            text=text,          # annotation text
            showarrow=False,    # remove arrow 
            xref='paper',     # use paper coords
            yref='paper',     #  for both coordinates
            xanchor='right',  # x-coord line up with right end of text 
            yanchor='bottom', # y-coord line up with bottom end of text 
            x=x,              # position's x-coord
            y=y               #   and y-coord
        )

    title = plot_title  # plot's title

    # Make a layout object
    layout1 = go.Layout(
        title=title,  # set plot's title
        font=go.layout.Font(
            family='PT Sans Narrow, sans-serif',  # global font
            size=13
        ),

        scene=dict(
            xaxis=dict(
                title='LatentX',
                #gridcolor='rgb(255, 255, 255)',
                #zerolinecolor='rgb(255, 255, 255)',
                #showbackground=True,
                #backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title='LatentY',
                #gridcolor='rgb(255, 255, 255)',
                #zerolinecolor='rgb(255, 255, 255)',
                #showbackground=True,
                #backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title='LatentZ',
                #gridcolor='rgb(255, 255, 255)',
                #zerolinecolor='rgb(255, 255, 255)',
                #showbackground=True,
                #backgroundcolor='rgb(230, 230,230)'
            ),
        ),    

        showlegend=True,  # remove legend
        # annotations= go.Annotations([  # add annotation citing the data source
        #     make_anno()
        # ]),
        #showlegend=True,  # remove legend
        autosize=False,    # custom size
        width=1200,         # set figure width 
        height=800,         #  and height
        margin=dict(l=0,
                    r=0,
                    b=30,
                    t=50
        )
    )


    legend=go.layout.Legend(
        x=0.9,
        y=0.8,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=16,
            color="black"
            ),
        bgcolor='rgba(0,0,0,0)',
        bordercolor="Black",
        borderwidth=2
        )
    
    x, y, z, label_id = ar_plot[:,0], ar_plot[:,1], ar_plot[:,2], ar_plot[:,3]

    # customize color
    mycolormapx =[]
    mypair = cl.scales['11']['qual']['Set3']
    mycolormap1 = cl.interp( mypair, 22 ) # Map color scale to 500 bins
    mycolormapx.append(mycolormap1)

    mypair = cl.scales['11']['qual']['Paired']
    mycolormap2 = cl.interp( mypair, 22 ) # Map color scale to 500 bins
    mycolormapx.append(mycolormap2)

    # flatten list of list to a single list
    mycolormap = [item for sublist in mycolormapx for item in sublist]
    
    # use default color map
    mycolormap = def_color * 10


    ls_symbol = [
        'circle', 'circle-open', 'square', 'square-open',
        'diamond', 'diamond-open', 'cross'] * 10
#     ls_symbol = ['circle', 'square', 'diamond'] * 5

    ls_classi = list(dic_labels.values())

    ######################################################
    trace = []
    for color_i, cat_i in enumerate(np.unique(label_id)):
        ar_label_indx = np.where(label_id == cat_i)[0]
#         import ipdb; ipdb.set_trace()
        trace0 = go.Scatter3d(
            x=x[ar_label_indx],
            y=y[ar_label_indx],
            z=z[ar_label_indx],
            mode='markers',
            marker=dict(
                size=ar_marker_size,
                # set color to an array/list of desired values
                color=mycolormap[color_i],
                #colorscale='Viridis',   # choose a colorscale Viridis
                #line = {'width' : 0.2, 'color' : 'rgba(255,255,255, 0.2)'},
                opacity=0.6,
                sizeref=20,
                sizemode='diameter',
                symbol=ls_symbol[color_i]
            ),
            text="class" + str(int(cat_i)).zfill(2),
            name=dic_labels[cat_i]  # "class" + str(int(cat_i)).zfill(2)
        )
        trace.append(trace0)
    ########################################################
    data = trace

    #
    layout = go.Layout(showlegend=True)

    fig = go.Figure(data=data, layout=layout1)
    #fig = go.Figure(data = data)
    fig['layout'].update({'legend' : legend})
    iplot(
        fig,
        config={
            'modeBarButtonsToRemove': ['sendDataToCloud'],
            'showLink': False, 'displaylogo' : False})
    #iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    #plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

    # save plot to html
    if len(file_dir) > 0:
#         filename = file_dir + '_{0}.html'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        fig.write_html(
            file_dir,
            config={
                'modeBarButtonsToRemove': ['sendDataToCloud'],
                'showLink': False, 'displaylogo' : False})

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
        'subtitle_size': 20,
        'subtitle_y': 0.9,
        'include_colorbar': True,
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
    subtitle_size = kwargs['subtitle_size']
    subtitle_y = kwargs['subtitle_y']
    include_colorbar = kwargs['include_colorbar']
    nrows = kwargs['nrows']
    ncols = kwargs['ncols']
   
    n_samples = len(X)
    n_images = nrows*ncols

    nrows = min(int(np.ceil(n_samples / ncols)), nrows)

    if set_random:
        if n_images < n_samples:
            idx_selected = np.random.choice(n_samples, nrows*ncols, replace=False)
        else:

            idx_selected = np.array(list(range(n_samples)))
            
    else:
        idx_selected = np.array(list(range(min(n_images, n_samples))))
    
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

    for i in range(min(n_samples, n_images)):
        ax = fig.add_subplot(nrows, ncols, i+1)
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
    
    if category is not None:
        fig.suptitle(
            "Class ID {0}".format(category),
            fontsize=subtitle_size,
            fontweight=0,
            color='black',
            style='italic',
            y=subtitle_y)
    if include_colorbar:
        cax = fig.add_axes([1, 0.1, 0.01, 0.8])
        fig.colorbar(im, cax=cax)
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


def robust_mean_repr(X, dl=0.05, ul=0.95):
    """
    calculate the robust mean of 2d array
    @X <3d array>: [nsamples, weight, height]
    """
    dl_ = np.quantile(X, [dl], axis=0)[0]
    ul_ = np.quantile(X, [ul], axis=0)[0]

    X_ = X.copy()
    for i in range(len(X)):
        X_[i][X[i] > ul_] = np.nan
        X_[i][X[i] < dl_] = np.nan
    return np.nanmean(X_, axis=0)


def plot_cluster_representation(arr_data, arr_labels, **kwargs):
    """
    plot cluster representation images by cluster
    @arr_data <3d array>: (nsamples, nkpis, nhours)
    @arr_label <1d array>: labels
    @method <str>: 'median', 'mean'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    def_kwargs = {
        'method': 'median',
        'ncols': 4,
        'bigfig_step': 1,  # the number of grids occupied by big image
        'yticklabels': '', 
        'suptitle': '',
        'y_suptitle': 0.92,
    }

    for k,v in def_kwargs.items():
        kwargs.setdefault(k, v)
    
    method = kwargs['method']
    ncols = kwargs['ncols']
    bigfig_step = kwargs['bigfig_step']
    yticklabels = kwargs['yticklabels']
    suptitle = kwargs['suptitle']
    y_suptitile = kwargs['y_suptitle']

    dic_method = {
        'median': np.median,
        'mean': np.mean
    }
    selcted_clusters = np.unique(arr_labels)
    
    types_ = list(set([type(s) for s in selcted_clusters]))
    
    if len(types_) == 1:
        try:
            selcted_clusters = sorted(selcted_clusters, key=int)
        except Exception:
            selcted_clusters = sorted(selcted_clusters)    
        
    lst_data_agg = []
    lst_cl_name = []
    lst_cl_support = []
    for i,v in enumerate(selcted_clusters):
        class_indx = np.where(arr_labels == v)[0]
        if method == 'robust_mean':
            lst_data_agg.append(robust_mean_repr((arr_data[class_indx])))
        else:
            lst_data_agg.append(dic_method[method](arr_data[class_indx], axis=0))
        lst_cl_name.append(v)
        lst_cl_support.append(len(class_indx))
    nrows = int(np.ceil(len(selcted_clusters) + bigfig_step*bigfig_step -1) / ncols)
    figsize = (ncols*5, nrows*5)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
#     for i,v in enumerate(selcted_clusters):
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.3)

    ix = 0

    ax = fig.add_subplot(gs[:bigfig_step, :bigfig_step])
    ax.imshow(
        lst_data_agg[ix],
        interpolation='spline16',
        cmap=plt.get_cmap('RdYlGn_r'))
    if len(yticklabels) > 0:
        ax.set_yticks(np.arange(len(yticklabels)))
        yticklabels = [
            "{}_{}".format(k_, i_) for( k_,i_) in zip(
                yticklabels, range(len(yticklabels)))]
        ax.set_yticklabels(yticklabels)
    ax.set_title("cluster: {}, #: {}".format(lst_cl_name[ix], lst_cl_support[ix]))

    ix += 1

    for row_ in range(bigfig_step):
        for col_ in range(bigfig_step, ncols):
            ax = fig.add_subplot(gs[row_, col_])
            ax.imshow(
                lst_data_agg[ix],
                interpolation='spline16',
                cmap=plt.get_cmap('RdYlGn_r'))
            ax.set_title("cluster: {}, #: {}".format(lst_cl_name[ix], lst_cl_support[ix]))
            ix += 1
            
            if ix >= len(selcted_clusters):
                break

    for row_ in range(bigfig_step, nrows):
        for col_ in range(ncols):
            ax = fig.add_subplot(gs[row_, col_])
            ax.imshow(
                lst_data_agg[ix],
                interpolation='spline16',
                cmap=plt.get_cmap('RdYlGn_r'))
            ax.set_title("cluster: {}, #: {}".format(lst_cl_name[ix], lst_cl_support[ix]))
            ix += 1
            
            if ix >= len(selcted_clusters):
                break

    if len(suptitle) > 0:
        fig.suptitle("{}, method={}".format(suptitle, method), y=y_suptitile)
