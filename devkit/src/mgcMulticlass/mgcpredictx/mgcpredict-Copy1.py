###----------------------------------------------------------------------------
#
# (c) Ericsson 2018 - All Rights Reserved
#
# No part of this material may be reproduced in any form
# without the written permission of the copyright owner.
# The contents are subject to revision without notice due 
# to continued progress in methodology, design and manufacturing. 
# Ericsson shall have no liability for any error or damage of any
# kind resulting from the use of these documents.
#
# Any unauthorized review, use, disclosure or distribution is 
# expressly prohibited, and may result in severe civil and 
# criminal penalties.
#
# Ericsson is the trademark or registered trademark of
# Telefonaktiebolaget LM Ericsson. All other trademarks mentioned
# herein are the property of their respective owners. 
# 
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

import sys
import os

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot 
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
#init_notebook_mode()

import colorlover as cl
import random
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import datetime, time
import scipy

__version__ = '0.0.18'

# Custom plot function ==============================


def mgcPlotKpibyClass(**kwargs):
    """
    return all kpi boxplot for each prediction class
    """
    # use default keyword parameters if user does not provide
    def_vals = {'plot_title' : None,
                'classx' : None,
                'df_input' : None,
                'kpi_list' : None,
                'top3_class' : None
               }

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)
        
    plot_title = kwargs['plot_title']
    classx = kwargs['classx']
    df_input = kwargs['df_input']
    kpi_list = kwargs['kpi_list']
    top3_class = kwargs['top3_class']
    

    # plot formatting ++++++++++++++++++++++++++++++++
    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">\
    echieng</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.Annotation(
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

    x_title = ''
    y_title = 'Dimension'

    font=dict(family='PT Sans Narrow, sans-serif', size=13)

    # Make a layout object
    layout1 = go.Layout(
        title = title,  # set plot's title
        font = dict(family='PT Sans Narrow, sans-serif', size=13),
        xaxis1 = dict(
            title = x_title,   # set x-axis title
            #range = xy_range,  # x-axis range
            zeroline = False   # remove x=0 line
        ),

        annotations = go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend = False,  # remove legend
        autosize = True,    # custom size
        #width = 980,         # set figure width 
        #height = 880,         #  and height
        margin = dict(l=100,
                    r=50,
                    b=180,
                    t=50
        )
    )
    #  ++++++++++++++++++++++++++++++++
    
    
    # data for plot +++++++++++++++++++++++++++++++++++
    traces = []

    arr_x = np.array(df_input[kpi_list])
    arr_x_mean = arr_x.reshape(-1, 24, 33).mean(axis = 1)

    itemindex = np.where(top3_class[:, 0]==classx)[0]
    print(classx)
    print("=============================================")
    kpi_set = arr_x_mean[itemindex, :]
    
    for i in range(len(kpi_list)):
        data_kpi_x = kpi_set[:, i]
        kpi_name = kpi_list[i]
        
        traces.append(go.Box(
            y = data_kpi_x,
            opacity=0.7,
            name=kpi_name
            
        ))        

    #   +++++++++++++++++++++++++++++++++++

    data = traces
    layout = go.Layout(
        yaxis=dict(
            title=y_title,
            zeroline=False,
            type='log',
            autorange=True
        ),
        xaxis = dict(tickangle = 90)
    )
    fig = go.Figure(data=data, layout=layout)

    fig.update(layout = layout1)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

def mgcPlotClassbyKpi(**kwargs):
    """
    
    """
    # use default keyword parameters if user does not provide
    def_vals = {'plot_title' : None,
                'kpi_col' : None,
                'df_input' : None,
                'pred_label' : None
               }

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)
        
    plot_title = kwargs['plot_title']
    kpi_col = kwargs['kpi_col']
    df_input = kwargs['df_input']
    pred_label = kwargs['pred_label']

    # plot formatting ++++++++++++++++++++++++++++++++
    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">\
    echieng</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.Annotation(
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

    x_title = ''
    y_title = 'Dimension'

    font=dict(family='PT Sans Narrow, sans-serif', size=13)

    # Make a layout object
    layout1 = go.Layout(
        title = title,  # set plot's title
        font = dict(family='PT Sans Narrow, sans-serif', size=13),
        xaxis1 = dict(
            title = x_title,   # set x-axis title
            #range = xy_range,  # x-axis range
            zeroline = False   # remove x=0 line
        ),

        annotations = go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend = False,  # remove legend
        autosize = True,    # custom size
        #width = 980,         # set figure width 
        #height = 880,         #  and height
        margin = dict(l=100,
                    r=50,
                    b=180,
                    t=50
        )
    )
    #  ++++++++++++++++++++++++++++++++
    
    
    # data for plot +++++++++++++++++++++++++++++++++++
    traces = []

    for _, classx in enumerate(pred_label):
        
        data_b1_kpi_0 = df_input[df_input['class1'] == classx]
        traces.append(go.Box(
            y = data_b1_kpi_0.groupby(['cell'])[kpi_col].max(),
            #x = class_id,
            opacity=0.7,
            name=classx
            
        ))
    #   +++++++++++++++++++++++++++++++++++

    data = traces
    layout = go.Layout(
        yaxis=dict(
            title=y_title,
            zeroline=False
        ),
        xaxis = dict(tickangle = 90)
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update(layout = layout1)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

def mgcPlotStack(pred_label = None, top3_class = None, nb_classes = None, **kwargs):
    """
    return stack bar plot
    
    Arguments:
        pred_label: prediction label
        top3_class: prediction array probability top 3
        nb_classes: number of classes for the label
    """
    
    bupu = cl.scales['9']['seq']['BuPu']
    bupu500 = cl.interp( bupu, 20 )    
    col = bupu500[:nb_classes]
    
    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">\
    echieng</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.Annotation(
            text=text,          # annotation text
            showarrow=False,    # remove arrow 
            xref='paper',     # use paper coords
            yref='paper',     #  for both coordinates
            xanchor='right',  # x-coord line up with right end of text 
            yanchor='bottom', # y-coord line up with bottom end of text 
            x=x,              # position's x-coord
            y=y               #   and y-coord
        )

    font = dict(family='PT Sans Narrow, sans-serif', size=13)
    title = 'Prediction Result <br>\
    Top2 Distribution'  # plot's title

    x_title = 'Top 2 Distribution'
    y_title = 'Top 2 Class'

    # Make a layout object
    layout1 = go.Layout(
        title = title,  # set plot's title
        font = dict(family='PT Sans Narrow, sans-serif', size=13),
        xaxis1 = dict(
            title = x_title,   # set x-axis title
            #range=xy_range,  # x-axis range
            zeroline=False   # remove x=0 line
        ),

        annotations = go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend = False,  # remove legend
        autosize = True,    # custom size
        #width=980,         # set figure width 
        #height=880,         #  and height
        margin = dict(l = 200,
                      r = 50,
                      b = 100,
                      t = 50
                     )
    )

    traces = []

    k=-1
    for i in pred_label:
        k = k+1
        current_distribution = np.zeros(len(pred_label), dtype=float)
        for j in range(len(pred_label)):
            # Obtain top 2 distribution
            top2_classes = top3_class[np.where(top3_class[:,0] == pred_label[j])[0],1]
            top2_unique_labels, top2_counts = np.unique(top2_classes, return_counts=True)
            top2_percentage = np.round(100 * (top2_counts / np.sum(top2_counts)), decimals=2)

            # Store results for current trace
            if(sum(top2_unique_labels == i) == 0):
                current_distribution[j] = 0
            else:
                current_distribution[j] = top2_percentage[top2_unique_labels == i][0]

        traces.append(go.Bar(
            x=current_distribution,
            y=pred_label,
            name=i,
            text=current_distribution,
            orientation = 'h',
            marker=dict(color=col[k], line=dict(color=col[k], width=1.5)),
            opacity=0.7
        ))

    data = traces
    layout = go.Layout(
        yaxis=dict(
            title=y_title,
            zeroline=False
        ),
        barmode='stack'
    )
    fig = go.Figure(data=data, layout=layout)

    fig.update(layout = layout1)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

def bar_dist_plot(pred_prob = None, dict_factor = None):
    """
    return bar plot
    
    Arguments:
        pred_prob: prediction probability
        dict_factor: dictionary factor for label
    """

    #unique_labels, counts = np.unique(top3_class[:,0], return_counts=True)
    unique_labels, counts = np.unique(np.argmax(pred_prob, axis = 1), return_counts = True)
    label_freq = pd.DataFrame({'Label':unique_labels,'Sample Counts':counts})
    label_freq['class'] = label_freq['Label'].map(dict_factor)
    label_freq['perc'] = label_freq['Sample Counts']/label_freq['Sample Counts'].sum()
    label_freq = label_freq.sort_values('perc')
    
    
    y = label_freq['class']
    x = label_freq['perc']
    x2 = label_freq['Sample Counts']
    
    bupu = cl.scales['9']['seq']['BuPu']
    bupu500 = cl.interp( bupu, 20 )

    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">\
    echieng</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.Annotation(
            text=text,          # annotation text
            showarrow=False,    # remove arrow 
            xref='paper',     # use paper coords
            yref='paper',     #  for both coordinates
            xanchor='right',  # x-coord line up with right end of text 
            yanchor='bottom', # y-coord line up with bottom end of text 
            x=x,              # position's x-coord
            y=y               #   and y-coord
        )

    font=dict(family='PT Sans Narrow, sans-serif', size=13)
    title = 'Network Overview<br>'
    x_title = 'Distribution' #.format(site1)  # x and y axis titles
    y_title = ''

    # Make a layout object
    layout1 = go.Layout(
        title = title,  # set plot's title
        font = dict(family='PT Sans Narrow, sans-serif', size=13),
        xaxis1 = dict(
            title = x_title,   # set x-axis title
            #range=xy_range,  # x-axis range
            zeroline=False   # remove x=0 line
        ),

        annotations = go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=False,  # remove legend
        autosize=True,    # custom size
        #width=980,         # set figure width 
        #height=880,         #  and height
        margin=dict(l=300,
                    r=50,
                    b=100,
                    t=50
        )
    )

    
    data = [go.Bar(
            y=y,
            x=x2,
            text = pd.Series(["{0:.2f}%".format(val * 100) for val in x]),
            textfont=dict(size=28, color='rgba(0, 0, 0, 1)'),
            textposition = 'auto', #'auto',
            orientation = 'h',
            marker = dict(
                color = bupu500,
                line = dict(color='rgb(140, 140, 140)', width=1)
            )
    )]

    layout = go.Layout(
        showlegend = False,
        autosize = True,
        xaxis = dict(
            title = x_title
        ),
        yaxis = dict(
            title = y_title,
            tickfont=dict(
            family='PT Sans Narrow, sans-serif',  # global font
            size=20
        )),
        hovermode='closest',
        bargap=2,

    )

    fig = go.Figure(data=data, layout=layout)
    fig.update(layout = layout1)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

def dens_hist_plot(**kwargs):
    """
    return prediction probability density histogram
    
    Arguments:
        df: classification prediction probability in pandas datafrane
    
    """
    

    
    data = {'top1prob' : random.sample(range(1, 100), 5),
            'top2prob' : random.sample(range(1, 100), 5)
           }
    
    def_vals = {"df" : data
           } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    df = kwargs['df']
    
    x = df[:,0]#df['top1prob']
    y = df[:,1]#df['top2prob']    
    
    # String citing the data source
    url='chin.lam.eng@ericsson.com'
    text_source = "Source and info: <a href=\"{}\">\
    echieng</a>".format(url)

    def make_anno(x=1, y=1, text=text_source):
        return go.Annotation(
            text=text,          # annotation text
            showarrow=False,    # remove arrow 
            xref='paper',     # use paper coords
            yref='paper',     #  for both coordinates
            xanchor='right',  # x-coord line up with right end of text 
            yanchor='bottom', # y-coord line up with bottom end of text 
            x=x,              # position's x-coord
            y=y               #   and y-coord
        )



    title = 'Prediction Result<br>\
    Top1, Top2'  # plot's title

    x_title = 'Top1 Probability'#.format(site1)  # x and y axis titles
    y_title = 'Top2 Probability'

    # Make a layout object
    layout1 = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow, sans-serif',  # global font
            size=13
        ),
        xaxis1=go.XAxis(
            title=x_title,   # set x-axis title
            #range=xy_range,  # x-axis range
            zeroline=False   # remove x=0 line
        ),

        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=True,  # remove legend
        autosize=False,    # custom size
        width=980,         # set figure width 
        height=880,         #  and height
        margin=dict(l=100,
                    r=50,
                    b=100,
                    t=50
        )
    )


    trace1 = go.Scatter(
        x=x, y=y, mode='markers', name='points',
        marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)
    )
    trace2 = go.Histogram2dcontour(
        x=x, y=y, name='density', ncontours=20,
        colorscale='Hot', reversescale=True, showscale=False
    )
    trace3 = go.Histogram(
        x=x, name='x density',
        marker=dict(color='rgb(102,0,0)'),
        yaxis='y2'
    )
    trace4 = go.Histogram(
        y=y, name='y density', marker=dict(color='rgb(102,100,200)'),
        xaxis='x2'
    )
    data = [trace1, trace2, trace3, trace4]

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        xaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False, title = x_title
        ),
        yaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False, title = y_title
        ),
        margin=dict(
            t=50
        ),
        hovermode='closest',
        bargap=0,
        xaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update(layout = layout1)
    
    
    fig['layout'].update(images= [dict(
              source= "image/0016_Blue_horizon.svg",
              xref= "paper",
              yref= "paper", xanchor="left", yanchor="bottom",
              x= 0,
              y= 0,
              sizex= 0.1,
              sizey= 0.1,
              sizing= "stretch",
              opacity= 0.5,
              layer= "above")])     

    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    plot(fig, filename='dens_hist.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})


def mgcTernaryScat(arr_top3_prob=None):

    """
    return ternary scatter plot for top 3 prediction probability
    
    Argument:
        arr_top3_prob: top 3 prediction probability array
    """
    

    import ternary
    ## Generate Data

    def generate_points(num_points=25, scale=100):
        points = []
        for i in range(num_points):
            x = arr_top3_prob[:,0]*100
            y = arr_top3_prob[:,1]*100
            z = scale - x - y
            points.append((x,y,z))
        return points

    ax_format = {'b' : "%.2f", 'l' : "%.2f", 'r' : "%.2f"}

    # Scatter Plot
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    figure.set_facecolor('white')

    ax = tax.get_axes()
    ax.patch.set_facecolor('white')

    # Set Axis labels and Title
    fontsize = 20
    tax.left_axis_label("Prediction Prob. 3", fontsize=fontsize)
    tax.right_axis_label("Prediction Prob. 2", fontsize=fontsize)
    tax.bottom_axis_label("Prediction Prob. 1", fontsize=fontsize)
    tax.bottom_axis_label("Prediction Prob. 1", fontsize=fontsize)

    tax.set_title("Scatter Plot", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="blue")
    
    # Plot a few different styles with a legend
    points = generate_points(30, scale=scale)
    tax.scatter(points, marker='s', color='red', label="Prediction Probability", alpha = 0.5)

    #points = generate_points(30, scale=scale)
    tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=10, tick_formats = ax_format)

    tax.clear_matplotlib_ticks()

    tax.show()
    
    #######################################################################################################################    
    
    
def mgcTernaryHeat(arr_top3_prob=None):
    """
    return ternary heatmap of top 3 prediction probability
    
    Argument:
        top 3 prediction probability array
    """
       

    import ternary
    ## Generate Data

    bin_cnt = 11
    scale = bin_cnt - 1
    
    def gen_heat(bin_cnt = 10):
        # generate heatmap
        ddd = np.histogram2d(arr_top3_prob[:, 0], arr_top3_prob[:, 1], bins=bin_cnt)
        
        # flatten density variable
        arr_ddd = np.array(pd.DataFrame(ddd[0]).stack().reset_index())
        
        # create tuple for density
        tup_list = list(zip(arr_ddd[:,0].astype(int), arr_ddd[:,1].astype(int), bin_cnt - (arr_ddd[:,0].astype(int)+ arr_ddd[:,1].astype(int))))

        #subset list of tuple
        tup_ind = []

        for i in ([item for item in tup_list if item[2] > 0]):
            tup_ind.append(tup_list.index(i))
        tup_list_sub = [tup_list[i] for i in tup_ind]  
        
        # generate dictionary
        d = dict(zip(tup_list, arr_ddd[:, 2]))
        # filter dictionary by a list of keys
        d_sub = dict(zip(tup_list_sub, [d[k] for k in tup_list_sub]))
        return d_sub
    
    d = gen_heat(bin_cnt)

    ax_format = {'b' : "%.1f", 'l' : "%.1f", 'r' : "%.1f"}

    # Scatter Plot

    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 7)
    figure.set_facecolor('white')

    ax = tax.get_axes()
    ax.patch.set_facecolor('white')

    colors=plt.cm.jet(np.linspace(0, 1, 3))

    # Set Axis labels and Title
    fontsize = 20

    tax.left_axis_label("Prediction 3", fontsize=fontsize)
    tax.right_axis_label("Prediction 2", fontsize=fontsize)
    tax.bottom_axis_label("Prediction 1", fontsize=fontsize)

    tax.set_title("Heatmap", fontsize=fontsize)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=1, color="blue")

    tax.heatmap(d, style="h", cmap=plt.get_cmap('viridis_r'))

    tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=1, tick_formats = ax_format)

    tax.clear_matplotlib_ticks()

    tax.show()
    
    #######################################################################################################################


def mgcHist2d(arr_top3_prob=None):
    """
    return 2d histogram of top 2 prediction probability
    
    Arguments:
        top 2 prediction probability array
    
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, MaxNLocator
    from numpy import linspace
    plt.ion()
     
    # Define a function to make the ellipses
    def ellipse(ra,rb,ang,x0,y0,Nb=100):
        xpos,ypos=x0,y0
        radm,radn=ra,rb
        an=ang
        co,si=np.cos(an),np.sin(an)
        the=linspace(0,2*np.pi,Nb)
        X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
        Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
        return X,Y
     
    # Define the x and y data 
    # For example just using random numbers
    x = arr_top3_prob[:, 0]
    y = arr_top3_prob[:, 1]
     
    # Set up default x and y limits
    xlims = [min(x),max(x)]
    ylims = [min(y),max(y)]
     
    # Set up your x and y labels
    xlabel = 'Prediction Prob. 1'
    ylabel = 'Prediction Prob. 2'
     
    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02
     
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
     
    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))
     
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram
     
    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
     
    # Find the min/max of the data
    xmin = min(xlims)
    xmax = max(xlims)
    ymin = min(ylims)
    ymax = max(y)
     
    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbins = 100
     
    xbins = linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
     
    H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
    X = xcenter
    Y = ycenter
    Z = H
     
    # Plot the temperature data
    cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))
     
    # Plot the temperature plot contours
    contourcolor = 'white'
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0
     
    X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
    axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points', horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25)
     
    X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
    axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)
     
    X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
    axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)
     
    #Plot the axes labels
    axTemperature.set_xlabel(xlabel,fontsize=25)
    axTemperature.set_ylabel(ylabel,fontsize=25)
     
    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')
     
    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')
     
    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)
     
    #Set up the histogram bins
    xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
    ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
     
    #Plot the histograms
    axHistx.hist(x, bins=xbins, color = 'blue')
    axHisty.hist(y, bins=ybins, orientation='horizontal', color = 'red')
     
    #Set up the histogram limits
    axHistx.set_xlim( min(x), max(x) )
    axHisty.set_ylim( min(y), max(y) )
     
    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')
     
    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')
     
    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))
     
    #Show the plot
    plt.draw()

    #######################################################################################################################
    
def mgcBoxclass(lst_plot, plot_title, pred_label=None, showflierx = False):
    """
    return box plot for each prediction class
    
    Arguments:
        list of key performance indicators associated with each prediction class
    """
    
    flierprops = dict(marker='o', markerfacecolor='red', markersize=3,
                      linestyle='None', markeredgewidth=0.0)
    # Multiple box plots on one Axes
    plt.rcParams["figure.figsize"] = (12,5)
    fig, ax = plt.subplots()
    box = ax.boxplot(lst_plot, 
                     vert = False, 
                     patch_artist = True, 
                     showmeans = True, 
                     meanline = True, 
                     showfliers = showflierx,
                     flierprops=flierprops, 
                     widths = 0.8)
    colors=plt.cm.viridis(np.linspace(0, 1, 14))

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # add axis texts
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_title(plot_title, fontsize=18)
    ax.set_yticklabels(pred_label, fontsize=12)
    plt.show()

    #######################################################################################################################  

    
def mgcBoxkpi(lst_classbykpi, plot_title, cols_feat):
    """
    return box plot for each key performance indicators
    
    Arguments:
        list of prediction class associated with each key performance indicator
    """
    
    flierprops = dict(marker='o', markerfacecolor='red', markersize=3,
                      linestyle='None', markeredgewidth=0.0)   
    # Multiple box plots on one Axes
    plt.rcParams["figure.figsize"] = (12,8)
    fig, ax = plt.subplots()
    #ax.boxplot(lst_classbykpi, vert=False, notch=True, patch_artist=True)
    box = ax.boxplot(lst_classbykpi, 
                     vert = False, 
                     patch_artist = True, 
                     showmeans = True, 
                     meanline = True, 
                     showfliers = False,
                     flierprops=flierprops, 
                     widths = 0.8)
    colors=plt.cm.viridis(np.linspace(0, 1, 33))

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    # add axis texts
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('KPI', fontsize=12)
    ax.set_title(plot_title, fontsize=18)
    ax.set_yticklabels(cols_feat, fontsize=12)
    plt.show()
    
    #######################################################################################################################         

def mgcDensityPlt(**kwargs):
    """
    return density plot for each class per KPI plot
    """
    def_vals = {'df_input' : None}

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    _df_input = kwargs['df_input']
    
    # custom color to avoid duplication for certain prediction category
    #colors = plt.cm.tab20(np.linspace(0, 1, 14))
    #colors = tuple(map(tuple, colors))
    
    colors = cm = plt.get_cmap('Set1')
    NUM_COLORS = 14
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)    
    
    # iterate through all the KPI columns
    for i, cols in enumerate(_df_input.columns[2:-1]):
        cols_target = ['class1'] + [cols]
        xx = _df_input[cols_target].groupby(['class1'])[cols]
        countx = 0
        plt.figure(figsize=(14,7))
        for p, q in xx:
            try:
                xx.apply(pd.DataFrame)[p].plot(kind='density',
                                                #color = colors[countx],
                                                color = (colors(countx//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS)),
                                                linestyle = LINE_STYLES[countx%NUM_STYLES])
                countx += 1
            except Exception:
                pass                
        plt.title("KPI # {0}".format(cols, color='red'), fontsize=18)
        plt.legend()
        plt.show()
    #######################################################################################################################         

def mgcDensityPlt2(**kwargs):
    """
    return density plot for each class per KPI
    """
    def_vals = {'df_input' : None}

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    _df_input = kwargs['df_input']
    from scipy import stats
    from matplotlib import gridspec
    colors=plt.cm.tab20(np.linspace(0, 1, 14))
    colors = tuple(map(tuple, colors))
    
    #colors = cm = plt.get_cmap('tab20')
    NUM_COLORS = 14
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    for i, cols in enumerate(_df_input.columns[2:-1]):
        cols_target = ['class1'] + [cols]
        xx = _df_input[cols_target].groupby(['class1'])[cols]
        label_uniq = np.sort((np.unique(_df_input['class1'])))[::-1]
        countx = 0
       
        df_xx = xx.apply(pd.DataFrame)
        
        # add noise to avoid singular matrix
        df_xx = df_xx.add(pd.DataFrame(0.000001*np.random.rand(df_xx.shape[0], df_xx.shape[1])), fill_value=0)

        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw = {'height_ratios':[5, 2]}, sharex = True, figsize=(12,7))
        fig.suptitle("KPI # {0}".format(cols, color='red'))
        fig.tight_layout()
        
        for p, q in xx:
            try:
                # add minor noise to avoid singular matrix
                x = np.array(df_xx[p])[~np.isnan(np.array(df_xx[p]))]
                x = np.sort(x, axis=None) 
                density = stats.kde.gaussian_kde(x)

                axes[0].plot(x, density(x), label = p,
                              color = colors[countx],
                              #color = (colors(countx//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS)),
                              linestyle = LINE_STYLES[countx%NUM_STYLES])
                axes[0].grid(linestyle = '-', color = 'white')
                #axes[0].set_facecolor('whitesmoke')
                axes[0].legend(loc='best', fancybox=True, framealpha=0.1)

                # for rug plot
                y2 = [-0.01*(countx + 1)]*len(x)
                axes[1].plot(x, y2, '|', 
                         #color=colors(countx//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS),
                         color = colors[countx])
                axes[1].grid(linestyle = '-', color = 'white')
                axes[1].set_yticklabels(label_uniq)
                countx += 1
            except Exception:
                pass           
            
        # set ticks visible, if using sharex = True. Not needed otherwise
        for tick in axes[0].get_xticklabels():
            tick.set_visible(True)
        plt.subplots_adjust(top=0.950)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    #######################################################################################################################     
    
    
def groupby_agg(**kwargs):
    
    def_vals = {'df_raw' : None,
                'kpi_list' : None,
                'statsopt' : 'max_min' }
    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    _df_input = kwargs['df_raw']
    kpi_list = kwargs['kpi_list'][:-1]
    statsopt = kwargs['statsopt']
    
    kpi_list_invert = ['LTE_CQI_Avg', 'LTE_User_THP_DL', 'LTE_User_THP_UL', 
                       'LTE_RaCbraSuccRatio','LTE_Avg_PUSCH_SINR', 'LTE_Avg_PUCCH_SINR']
    
    kpi_list_invert = list(set(kpi_list).intersection(kpi_list_invert)).copy()
    
    # kpi columns to numeric
    #_df_input = df_raw.copy()
    _df_input[kpi_list] = _df_input[kpi_list].apply(pd.to_numeric, errors='coerce')

    # default whole kpi_list aggregate
    dict_togroup = dict(zip(kpi_list, ['max']*len(kpi_list)))
    
    # update default dictionary with specific kpi aggregrate by minimum
    dict_togroup.update(dict(zip(kpi_list_invert, ['min']*len(kpi_list_invert))))
    
    # aggregate method
    if statsopt == 'max_min':      
        # group dataframe using the aggregation criteria in dict_togroup
        df_pred_agg = _df_input[['cell', 'class1'] + kpi_list].groupby(['cell', 'class1']).agg(dict_togroup)
    
    elif statsopt == 'median':
        df_pred_agg = _df_input[['cell', 'class1'] + kpi_list].groupby(['cell', 'class1'])[kpi_list].median()
    
    elif statsopt == 'average':
        df_pred_agg = _df_input[['cell', 'class1'] + kpi_list].groupby(['cell', 'class1'])[kpi_list].mean()
        
    
    # reset the multi-index colum to flat
    df_pred_agg.reset_index(inplace=True)

    # filtered out low sample class so that kde works without error
    filt = df_pred_agg.groupby(['class1'])['LTE_Active_DL_Users_TTI'].filter(lambda x: len(x) >= 3)
    df_final = df_pred_agg[df_pred_agg['LTE_Active_DL_Users_TTI'].isin(filt)]
    
    # rearrange the columns sequence
    df_final = df_final[['cell', 'class1'] + kpi_list]
    
    return df_final    
    #######################################################################################################################         
    
# Check value range / whole number for current data set
def kpiDesc(feat_data, kpi_list, nb_hours, nb_kpis):
        current_data = feat_data.reshape(feat_data.shape[0], nb_hours, nb_kpis)
        print('min-max normalized distribution')
        arr_x = current_data.reshape(-1, current_data.shape[2])
        pd.DataFrame(arr_x, columns = kpi_list).hist(figsize=(26, 16), normed = True)
        plt.show()
        
    
# Class ==============================

class mgcPredict(object):
    """
    predict cell issue classification given the key performance indicators features
    
    Arguments:
        pred_label: cell issue classification
        kpi_list: the key performance indicator follow the sequence
        freq_bandwidth: 10MHz, 15MHz, 20MHz
        dict_kpi_encode: optional kpi value for normalization
        dict_kpi_weight: optional kpi weight factor
        this is local version 0.0.18
        
    """
    
    def __init__(self, **kwargs):

        # create directory if does not exist
        os.makedirs('output/model/', exist_ok=True)
        os.makedirs('output/graph/', exist_ok=True)

        
        ####
        # Class names
        pred_label = ['Cell_Load',
                       'Cell_Load_Interference',
                       'Coverage_DTX',
                       'High_DL_UL_Utilization',
                       'High_DL_Utilization',
                       'High_UL_Utilization',
                       'Normal', 
                       'PUCCH_Interference',
                       'PUSCH_Interference',
                       'PUSCH_PUCCH_Interference',
                       'Poor_PUCCH_Performance',
                       'Processor_Load',
                       'RACH_Access',
                       'Signaling_Load']
        
        
        kpi_list = ["LTE_Active_DL_Users_TTI", 
                    "LTE_Active_UL_Users_TTI", 
                    "LTE_SE_DL_TTI", 
                    "LTE_SE_UL_TTI", 
                    "LTE_RRC_Conn_Att", 
                    "LTE_RRC_Connected_Users", 
                    "LTE_Max_RRC_Connetecd_Users", 
                    "LTE_DataVol_DL_DRB", 
                    "LTE_DataVol_UL_DRB", 
                    "LTE_DataVol_DL_SRB", 
                    "LTE_DataVol_UL_SRB", 
                    "LTE_PrbUtilDl_Avg", 
                    "LTE_PrbUtilUl_Avg", 
                    "LTE_PDCCH_CCE_Load_Avg", 
                    "LTE_CQI_Avg", 
                    "LTE_CQI_06_Rate", 
                    "LTE_User_THP_DL", 
                    "LTE_User_THP_UL", 
                    "LTE_RaCbra_Att", 
                    "LTE_RaCbraSuccRatio", 
                    "LTE_Average_HARQ_DTX_Ratio_DL", 
                    "LTE_Average_HARQ_DTX_Ratio_UL", 
                    "LTE_PUSCH_SINR_NEG2DB_RATE", 
                    "LTE_PUCCH_SINR_0DB_RATE", 
                    "LTE_Avg_PUSCH_SINR", 
                    "LTE_Avg_PUCCH_SINR", 
                    "LTE_UL_RSSI_PUSCH", 
                    "LTE_UL_RSSI_PUCCH", 
                    "LTE_UL_PATHLOSS_BELOW_130dB_RATE", 
                    "LTE_Avg_UL_PATHLOSS", 
                    "LTE_UE_Power_Limited_Ratio", 
                    "LTE_Avg_MP_Load", 
                    "LTE_90Perc_MP_Load"]

        self.nb_classes = len(pred_label)
        # Image size
        self.nb_hours = 24
        self.nb_kpis = len(kpi_list)
       

        #### set default keyword parameter value, else update with user input params
        def_vals = {"pred_label": pred_label,
                    "kpi_list": kpi_list,
                    "freq_bandwidth": 20}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        self.pred_label = kwargs['pred_label']
        self.kpi_list = kwargs['kpi_list']
        self.freq_bandwidth = kwargs['freq_bandwidth']

        # dictionary from list
        self.dict_label = {j: i for i, j in enumerate(self.pred_label)}

        # inverse label
        self.dict_factor = {v: k for k, v in self.dict_label.items()}

        # Colors       
        bupu = cl.scales['9']['seq']['BuPu']
        bupu500 = cl.interp( bupu, 16 ) 
        col = bupu500[:self.nb_classes]

    
    def mgcLoadModel(self, **kwargs):
        """
        load pre-trained model
        """
        #### set default keyword parameter value, else update with user input params
        def_vals = {"model_name": False}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        model_name = kwargs['model_name']
        
        if model_name:
            filepath = os.path.join(os.path.dirname(__file__), 'resources', model_name)
        else:
            filepath = os.path.join(os.path.dirname(__file__), 'resources', 'best_model_0907.hdf5')
        # Load prediction model
        print(filepath)
        self.cnn_model = load_model(filepath)
        return self

    def mgcLoadData(self, filepath = 'data/cleaned_data2.csv', **kwargs):
        """
        load input data
        
        """

        normalization = 'Regular' #'Strict', 'Regular'
        df = pd.read_csv(filepath)
        col_head = self.kpi_list
        df_raw = df.loc[:, ['cell', 'hour'] + col_head]
        df_raw = df_raw.sort_values(by = ['cell', 'hour'])
        df_raw.hour = df_raw.hour.apply(str).str.zfill(2)
        self.df_raw = df_raw
        
        return self        
          
    def mgcLoadCodebook(self, **kwargs):
        """
        load the kpi normalization codebook
        
        """

        dict_kpi_max = {'LTE_90Perc_MP_Load': 100,
                         'LTE_Average_HARQ_DTX_Ratio_DL': 100,
                         'LTE_Average_HARQ_DTX_Ratio_UL': 100,
                         'LTE_Avg_MP_Load': 100,
                         'LTE_Avg_UL_PATHLOSS': 140,
                         'LTE_CQI_06_Rate': 100,
                         'LTE_CQI_Avg': 15,
                         'LTE_PDCCH_CCE_Load_Avg': 100,
                         'LTE_PUCCH_SINR_0DB_RATE': 100,
                         'LTE_PUSCH_SINR_NEG2DB_RATE': 100,
                         'LTE_PrbUtilDl_Avg': 200,
                         'LTE_PrbUtilUl_Avg': 100,
                         'LTE_RaCbraSuccRatio': 100,
                         'LTE_UE_Power_Limited_Ratio': 100,
                         'LTE_UL_PATHLOSS_BELOW_130dB_RATE': 100}

        dict_kpi_min = {'LTE_UL_RSSI_PUCCH': -97,
                        'LTE_UL_RSSI_PUSCH': -97}
                        
        
        dict_kpi_encode = {'LTE_90Perc_MP_Load': 70,
                 'LTE_Active_DL_Users_TTI': 4,
                 'LTE_Active_UL_Users_TTI': 4,
                 'LTE_Average_HARQ_DTX_Ratio_DL': 50,
                 'LTE_Average_HARQ_DTX_Ratio_UL': 50,
                 'LTE_Avg_MP_Load': 50,
                 'LTE_Avg_PUCCH_SINR': 4.5,
                 'LTE_Avg_PUSCH_SINR': 20,
                 'LTE_Avg_UL_PATHLOSS': 140,
                 'LTE_CQI_06_Rate': 40,
                 'LTE_CQI_Avg': 15,
                 'LTE_DataVol_DL_DRB': 30000000,
                 'LTE_DataVol_DL_SRB': 50000,
                 'LTE_DataVol_UL_DRB': 10000000,
                 'LTE_DataVol_UL_SRB': 10000,
                 'LTE_Max_RRC_Connetecd_Users': 500,
                 'LTE_PDCCH_CCE_Load_Avg': 50,
                 'LTE_PUCCH_SINR_0DB_RATE': 10,
                 'LTE_PUSCH_SINR_NEG2DB_RATE': 60,
                 'LTE_PrbUtilDl_Avg': 70,
                 'LTE_PrbUtilUl_Avg': 70,
                 'LTE_RRC_Conn_Att': 15000,
                 'LTE_RRC_Connected_Users': 80,
                 'LTE_RaCbraSuccRatio': 100,
                 'LTE_RaCbra_Att': 10000,
                 'LTE_SE_DL_TTI': 5,
                 'LTE_SE_UL_TTI': 5,
                 'LTE_UE_Power_Limited_Ratio': 100,
                 'LTE_UL_PATHLOSS_BELOW_130dB_RATE': 100,
                 'LTE_UL_RSSI_PUCCH': -100,
                 'LTE_UL_RSSI_PUSCH': -100,
                 'LTE_User_THP_DL': 30,
                 'LTE_User_THP_UL': 15}

        # Min. encoding value definition
        dict_norm_min = {'LTE_Active_DL_Users_TTI': 0,
                     'LTE_Active_UL_Users_TTI': 0,
                     'LTE_SE_DL_TTI': 0,
                     'LTE_SE_UL_TTI': 0,
                     'LTE_RRC_Conn_Att': 0,
                     'LTE_RRC_Connected_Users': 0,
                     'LTE_Max_RRC_Connetecd_Users': 0,
                     'LTE_DataVol_DL_DRB': 0,
                     'LTE_DataVol_UL_DRB': 0,
                     'LTE_DataVol_DL_SRB': 0,
                     'LTE_DataVol_UL_SRB': 0,
                     'LTE_PrbUtilDl_Avg': 0,
                     'LTE_PrbUtilUl_Avg': 0,
                     'LTE_PDCCH_CCE_Load_Avg': 0,
                     'LTE_CQI_Avg': 0,
                     'LTE_CQI_06_Rate': 0,
                     'LTE_User_THP_DL': 0,
                     'LTE_User_THP_UL': 0,
                     'LTE_RaCbra_Att': 0,
                     'LTE_RaCbraSuccRatio': 0,
                     'LTE_Average_HARQ_DTX_Ratio_DL': 0,
                     'LTE_Average_HARQ_DTX_Ratio_UL': 0,
                     'LTE_PUSCH_SINR_NEG2DB_RATE': 0,
                     'LTE_PUCCH_SINR_0DB_RATE': 0,
                     'LTE_Avg_PUSCH_SINR': 0,
                     'LTE_Avg_PUCCH_SINR': 0,
                     'LTE_UL_RSSI_PUSCH': -120,
                     'LTE_UL_RSSI_PUCCH': -120,
                     'LTE_UL_PATHLOSS_BELOW_130dB_RATE': 0,
                     'LTE_Avg_UL_PATHLOSS': 80,
                     'LTE_UE_Power_Limited_Ratio': 0,
                     'LTE_Avg_MP_Load': 20,
                     'LTE_90Perc_MP_Load': 0}         

        # default weight factor
        dict_kpi_weight = dict(zip(dict_kpi_encode.keys(), [0.5] * self.nb_kpis))        

        #### set default keyword parameter value, else update with user input params
        def_vals = {"dict_kpi_encode": dict_kpi_encode,
                    "dict_kpi_weight": dict_kpi_weight,
                    "quantile_val": 0.99}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        self.dict_kpi_encode = kwargs['dict_kpi_encode']
        self.dict_kpi_weight = kwargs['dict_kpi_weight']
        quantile_val = kwargs['quantile_val']
        self.dict_norm_min = dict_norm_min
            
        def update_dict_x(dict_kpi_x):
            # update the parameter value with weight factor
            dict_kpi_x.update({n: dict_kpi_x[n]/(0.5 + self.dict_kpi_weight[n]) for n in self.dict_kpi_weight.keys()})
            # update parameter for max 
            dict_kpi_x.update({n: min(dict_kpi_x[n], dict_kpi_max[n]) for n in dict_kpi_max.keys()})
            # update parameter for negative max
            dict_kpi_x.update({n: max(dict_kpi_x[n], dict_kpi_min[n]) for n in dict_kpi_min.keys()})
            # round to nearest integer
            dict_kpi_x.update({n: round(dict_kpi_x[n], 0) for n in dict_kpi_x.keys()})
            return dict_kpi_x            

        # update dict_kpi_encode with final weight
        self.dict_kpi_encode = update_dict_x(self.dict_kpi_encode)
        self.norm_thresh = self.dict_kpi_encode
        #print(self.dict_kpi_encode)
        
        # quantile value
        self.dict_kpi_quantile = dict(zip(np.transpose(self.df_raw.quantile(quantile_val)).keys(), np.transpose(self.df_raw.quantile(quantile_val)).values))
        self.dict_kpi_quantile = update_dict_x(self.dict_kpi_quantile)
        dict_kpi_min_pos = {'LTE_90Perc_MP_Load': 10}
        # update parameter for max 
        self.dict_kpi_quantile.update({n: max(self.dict_kpi_quantile[n], dict_kpi_min_pos[n]) for n in dict_kpi_min_pos.keys()})
        
        return self

        
    def mgcNormData(self, **kwargs):
        #### set default keyword parameter value, else update with user input params
        def_vals = {"kpi_quantile": False}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        self.kpi_quantile = kwargs['kpi_quantile']
        
        df_raw_norm = self.df_raw.copy()
        
        if self.kpi_quantile:
            _norm_thresh = self.dict_kpi_quantile
        else:
            _norm_thresh = self.norm_thresh
    
        for i, j in enumerate(self.kpi_list):
            df_raw_norm[j] = (df_raw_norm[j] - self.dict_norm_min[j]) / (_norm_thresh[j] - self.dict_norm_min[j])
            # Cap all values below 0 and above 1
            df_raw_norm[j] = df_raw_norm[j].clip(0,1)

        # invert kpi value los as red colour in heatmap
        kpi_list_invert = ['LTE_CQI_Avg', 'LTE_User_THP_DL', 'LTE_User_THP_UL', 
                           'LTE_RaCbraSuccRatio','LTE_Avg_PUSCH_SINR', 'LTE_Avg_PUCCH_SINR']
            
        for i, j in enumerate(kpi_list_invert):
            df_raw_norm[j] = 1 - df_raw_norm[j]
        
        df_raw_norm.sort_values(by = ['cell', 'hour'], inplace=True)
        #print(df_raw_norm.columns)

        # transform to wide table
        df_raw_norm_x_w = pd.DataFrame(df_raw_norm.groupby(['cell', 'hour']).sum().unstack('hour'))
   
        # drop nan rows
        df_raw_norm_x_w.dropna(how='any', inplace = True)
        
        # rename the columns and pattern the hierarchical header
        new_cols = [''.join(t) for t in df_raw_norm_x_w.columns]
        df_raw_norm_x_w.columns = new_cols
        df_raw_norm_x_w.reset_index(level = 0, inplace=True)            
        df_raw_norm_x_w.index = df_raw_norm_x_w['cell']

        
        _df_raw = self.df_raw.copy()
        df_raw_x_w = pd.DataFrame(_df_raw.groupby(['cell', 'hour']).sum().unstack('hour'))        
        df_raw_x_w.dropna(how='any', inplace = True)
        df_raw_x_w.reset_index(inplace=True)        
        # rename the columns and pattern the hierarchical header
        new_cols = [''.join(t) for t in df_raw_x_w.columns]
        df_raw_x_w.columns = new_cols
        df_raw_x_w.reset_index(level = 0, inplace=True)        
        df_raw_x_w.index = df_raw_x_w['cell']

        
        # rearrange columns by kpi and followed by hour
        new_list = []
        for i in range(24):
            new_list.extend([s + str(i).zfill(2) for s in self.kpi_list])

        """
        # rearrange columns by hour then kpi
        new_list = []
        for i in range(len(self.kpi_list)):
            new_list.extend(self.kpi_list[i] + str(s).zfill(2) for s in range(24))"""
            
        df_raw_norm_x_w = df_raw_norm_x_w[new_list]
        df_raw_x_w = df_raw_x_w[new_list]
        
        self.df_raw_norm_x_w = df_raw_norm_x_w
        self.df_raw_x_w = df_raw_x_w
        
        feat_dataset = np.array(df_raw_norm_x_w.iloc[:, :])
        self.feat_dataset = feat_dataset.astype('float32')
        
        feat_dataset_raw = np.array(df_raw_x_w.iloc[:, :])
        self.feat_dataset_raw = feat_dataset_raw.astype('float32')        
        
        #self.df_raw_norm = df_raw_norm_x_w
        self.feat_norm = self.feat_dataset
        
        return self
        



    def mgcStatsDesc(self, **kwargs):
        kpiDesc(self.feat_dataset, self.kpi_list, self.nb_hours, self.nb_kpis)
        
        print('kpi distribution')
        self.df_raw[self.kpi_list].hist(figsize=(26, 16), normed = True)
        plt.show()
        return self
        
    def mgcAutoencoder(self, **kwargs):
        """
        autoencoder
        """
        from keras.layers import Input, Dense
        from keras.models import Model
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        import numpy as np

        # Some memory clean-up
        from keras import backend as K
        K.clear_session()
        
        def_vals = {"x_train" : self.feat_norm,
                    "epochs" : 50,
                    "batch_size" : 64,
                    "encoding_dim": 20
                   }


        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        x_train = kwargs['x_train']
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.encoding_dim = kwargs['encoding_dim']
        
        # Network Architecture
        # this is the size of our encoded representations
        encoding_dim = self.encoding_dim  # 
        img_shp = np.array(x_train).shape[1] # number of columns

        # this is our input placeholder
        input_img = Input(shape=(img_shp,), name="input_ae")
        # "encoded" is the encoded representation of the input
        encoded = Dense(500, activation='relu', name="dense_ae1")(input_img)
        encoded = Dense(100, activation='relu', name="dense_ae1a")(encoded)
        encoded = Dense(encoding_dim, activation='relu', name="dense_ae2")(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(500, activation='relu', name="dense_ae3")(encoded)
        decoded = Dense(100, activation='relu', name="dense_ae3a")(decoded)
        decoded = Dense(img_shp, activation='linear', name="dense_ae4")(decoded)

        # this model maps an input to its reconstruction
        autoencoder_d = Model(input_img, decoded)
        
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder_d.layers[-2]

        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        
        print('autoencoder')
        autoencoder_d.summary()
        
        print('encoder')
        encoder.summary()
        
        # plot generator model
        plot_model = SVG(model_to_dot(autoencoder_d, show_shapes = True).create(prog='dot', format='svg'))   
        
        # compile
        autoencoder_d.compile(loss='mean_squared_error', optimizer='adam')
        
        # train
        autoencoder_d.fit(np.array(x_train), 
                          np.array(x_train),
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          shuffle=True,
                          validation_split=0.2)
        
        # encoder output
        self.encoded_imgs = encoder.predict(np.array(x_train))
        
        self.autoencoder_d = autoencoder_d
        self.encoder = encoder
        self.decoder = decoder
        self.plot_model = plot_model

        return self
    #############################################################################################    

    def mgcReconstructLoss(self, **kwargs):
        """
        calculate reconstruction error
        """
        def_vals = {"x_train" : self.feat_norm
                   }


        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        x_train = kwargs['x_train']
        
        from keras import backend as K
        
        # prediction reconstruction output to find mean squared error against original input variables
        _x_pred = self.autoencoder_d.predict(np.array(x_train), batch_size = self.batch_size)

        # function to calculate means square error
        def mean_squared_error(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)

        # calculate reconstruction means square error
        mse = mean_squared_error(x_train, _x_pred)

        # transform tensor to numpy using tensorflow .eval() function

        import tensorflow as tf
        sess = tf.InteractiveSession()
        self.arr_mse = mse.eval()
        sess.close()
        return self
    ############################################################################################# 
        
    def mgcPred(self, **kwargs):
        """
        
        """

        def mgcAutoencoder_(**kwargs):
            """
            autoencoder
            """
            from keras.layers import Input, Dense
            from keras.models import Model
            from IPython.display import SVG
            from keras.utils.vis_utils import model_to_dot
            import numpy as np

            # Some memory clean-up
            from keras import backend as K
            K.clear_session()
            
            def_vals = {"x_train" : self.feat_norm,
                        "epochs" : 50,
                        "batch_size" : 64,
                        "encoding_dim": 5
                       }


            for k, v in def_vals.items():
                kwargs.setdefault(k, v)

            x_train = kwargs['x_train']
            epochs = kwargs['epochs']
            batch_size = kwargs['batch_size']
            encoding_dim = kwargs['encoding_dim']
            
            # Network Architecture
            # this is the size of our encoded representations
            encoding_dim = encoding_dim  # 
            img_shp = np.array(x_train).shape[1] # number of columns

            # this is our input placeholder
            input_img = Input(shape=(img_shp,), name="input_ae")
            # "encoded" is the encoded representation of the input
            encoded = Dense(100, activation='relu', name="dense_ae1")(input_img)
            encoded = Dense(encoding_dim, activation='relu', name="dense_ae2")(encoded)

            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(100, activation='relu', name="dense_ae3")(encoded)
            decoded = Dense(img_shp, activation='linear', name="dense_ae4")(decoded)

            # this model maps an input to its reconstruction
            autoencoder_d = Model(input_img, decoded)
            
            # this model maps an input to its encoded representation
            encoder = Model(input_img, encoded)

            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(encoding_dim,))
            # retrieve the last layer of the autoencoder model
            decoder_layer = autoencoder_d.layers[-2]

            # create the decoder model
            decoder = Model(encoded_input, decoder_layer(encoded_input))
            
            #print('autoencoder')
            #autoencoder_d.summary()
            
            #print('encoder')
            #encoder.summary()
            
            # plot generator model
            #plot_model = SVG(model_to_dot(autoencoder_d, show_shapes = True).create(prog='dot', format='svg'))   
            
            # compile
            autoencoder_d.compile(loss='mean_squared_error', optimizer='adam')
            
            # train
            autoencoder_d.fit(np.array(x_train), 
                              np.array(x_train),
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=0)
            
            # encoder output
            encoded_imgs = encoder.predict(np.array(x_train))


            return autoencoder_d
        #############################################################################################    

        def mgcReconstructLoss_(**kwargs):
            """
            calculate reconstruction error
            """
            def_vals = {"x_train" : self.feat_norm,
                        "batch_size" : 64,
                        "autoencoder_d" : None
                       }


            for k, v in def_vals.items():
                kwargs.setdefault(k, v)

            x_train = kwargs['x_train']
            batch_size = kwargs['batch_size']
            autoencoder_d = kwargs['autoencoder_d']
            
            from keras import backend as K
            
            # prediction reconstruction output to find mean squared error against original input variables
            _x_pred = autoencoder_d.predict(np.array(x_train), batch_size = batch_size)

            # function to calculate means square error
            def mean_squared_error(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true), axis=-1)

            # calculate reconstruction means square error
            mse = mean_squared_error(x_train, _x_pred)

            # transform tensor to numpy using tensorflow .eval() function

            import tensorflow as tf
            sess = tf.InteractiveSession()
            arr_mse = mse.eval()
            sess.close()
            return arr_mse
        ############################################################################################# 
        
        
        # Obtain prediction probabilities for whole dataset
        start = time.time()

        # Create 4D tensor
        #print(self.feat_norm.shape)
        feat_dataset_norm_ = self.feat_norm.reshape(-1, self.nb_hours, self.nb_kpis, 1)

        # Data augmentation was used for the prediction model, hence enlarge images
        feat_dataset_norm_ = scipy.ndimage.zoom(feat_dataset_norm_, (1, 72/self.nb_hours, 72/self.nb_kpis, 1), order=0)

        # Prediction
        start = time.time()
        self.prediction_prob = self.cnn_model.predict(feat_dataset_norm_)
        end = time.time()
        print("Prediction Execution time [min]: " + str((end - start)/60))
        
        # Anomaly
        start = time.time()
        ae_d = mgcAutoencoder_()
        self.arr_mse = mgcReconstructLoss_(autoencoder_d = ae_d)
        end = time.time()
        print("Anomaly Execution time [min]: " + str((end - start)/60))        
        
        # Obtain top 3 indices of the prediction probability
        top3_idx = np.argsort(self.prediction_prob, axis=1)[:,::-1][:,:3]

        # Top 3 class & prediction probability 
        top3_class = np.transpose(np.repeat(self.pred_label, self.prediction_prob.shape[0], axis=0).reshape(len(self.pred_label),self.prediction_prob.shape[0]))
        self.top3_class = top3_class[np.repeat(np.arange(self.prediction_prob.shape[0]), 3), top3_idx.ravel()].reshape(self.prediction_prob.shape[0], 3)
        self.top3_pred = self.prediction_prob[np.repeat(np.arange(self.prediction_prob.shape[0]), 3), top3_idx.ravel()].reshape(self.prediction_prob.shape[0], 3)

        return self
    
    def mgcViz(self, **kwargs):

        self.plotBar = bar_dist_plot(pred_prob = self.prediction_prob, dict_factor = self.dict_factor)
        print('')
        print('total number of {0: .0f} unique cell, day'.format(len(self.df_raw['cell'].unique())))
        
        return self       
    
    def mgcDensHist(self, **kwargs):
        """
        density histogram
        """
        
        self.plotDensHist = dens_hist_plot(df = self.top3_pred[:,:2])
        
        return self

    def mgcStackBar(self, **kwargs):
        """
        return stacked bar plot
        """
        
        self.plotStackBar = mgcPlotStack(pred_label = self.pred_label, top3_class = self.top3_class, nb_classes = self.nb_classes)
        
        return self      
    
    def mgcWrite(self, **kwargs):
        """
        return prediction joined with cell key performance indicators
        """
        def_vals = {"filepath" : 'output/cell_prediction.csv'
                   }


        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        filepath = kwargs['filepath']
        
        # cell id and kpi_hour columns only
        df_kpi_cell = self.df_raw_x_w 
        
        nb_hour = 24
        nb_kpi = 33
        
        # to numpy array
        arr_kpi = df_kpi_cell.values.reshape(df_kpi_cell.shape[0], nb_hour, nb_kpi)
        arr_cellid = df_kpi_cell.index.values.reshape(df_kpi_cell.shape[0], 1)
        #arr_pred = np.concatenate((self.top3_class, self.top3_pred), axis =1)
        arr_pred = np.concatenate((self.top3_class, self.top3_pred, self.arr_mse.reshape(self.arr_mse.shape[0], 1)), axis =1)

        list_of_arr = [] # initiate an empty list
        arr_hour = np.arange(nb_hour).reshape(nb_hour, 1) # 24 hours

        # timer
        start = time.time()

        imax = len(arr_kpi) # number of cells

        # iterate each index of individual array

        for i in range(imax):
            arr_kpi_i = arr_kpi[i]
            arr_cell_i = np.tile(arr_cellid[i], nb_hour)[None].T # complete 24 rows
            arr_pred_i = np.tile(arr_pred[i], (nb_hour, 1)) # complete 24 rows
            arr_temp = np.hstack([arr_cell_i , arr_hour, arr_kpi_i, arr_pred_i]) # combine arrays into one array
            list_of_arr.append(arr_temp)

        list_of_arr = np.asarray(list_of_arr) # list of arrays to array

        col_head = ['cell', 'hour'] + self.kpi_list + ['class1', 'class2', 'class3', 'prob1', 'prob2', 'prob3', 'mse']

        df_kpi_cell_pred = pd.DataFrame(np.vstack(list_of_arr), columns = col_head) # to pandas

        end = time.time()
        print("Execution time [sec]: " + str((end - start)/1))
        print(len(list_of_arr))
        print(df_kpi_cell_pred.shape)
        print(df_kpi_cell_pred.head())
        df_kpi_cell_pred.iloc[:,:].to_csv(filepath, index = False)
        print('export done!')
        self.df_kpi_cell_pred = df_kpi_cell_pred
        return self
            
    def mgcClassbyKpi(self, **kwargs):
        """
        return boxplot of each class
        """
        def_vals = {'kpi_list' : self.kpi_list[:3]}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        kpi_list = kwargs['kpi_list']
        
        
        for _, kpix in enumerate(kpi_list):
            plot_titlex = str(kpix)
            #kpi_col = kpix
            mgcPlotClassbyKpi(plot_title = plot_titlex, 
                              kpi_col = kpix, 
                              df_input = self.df_kpi_cell_pred, 
                              pred_label = self.pred_label)
        return self

    def mgcKpibyClass(self, **kwargs):
        """
        return boxplot of each class
        """
        def_vals = {'pred_label' : self.pred_label[:3],
                    'df_input' : self.df_kpi_cell_pred}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        pred_label = kwargs['pred_label']
        _df_input = kwargs['df_input']

        for _, predx in enumerate(pred_label):
            plot_titlex = str(predx)
            params = {'plot_title' : plot_titlex,
                      'classx' : predx,
                      'df_input' : _df_input,
                      'kpi_list' : self.kpi_list,
                      'top3_class' : self.top3_class}
            mgcPlotKpibyClass(**params)
        return self
        
    def mgcKpibyClassMat(self, **kwargs):
        """
        return boxplot of each class
        """
        def_vals = {'pred_label' : self.pred_label[:3],
                    'df_input' : self.df_kpi_cell_pred,
                    'cols_feat' : self.kpi_list, 
                    'statsopt' : 'max'}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        pred_label = kwargs['pred_label']
        _df_input = kwargs['df_input']
        cols_feat = kwargs['cols_feat']
        statsopt = kwargs['statsopt']

        # kpi columns to numeric
        _df_input[cols_feat] = _df_input[cols_feat].apply(pd.to_numeric, errors='coerce')
        
        lst_kpibyclass = []
        if statsopt == 'max':
            for i, j in enumerate(pred_label):
                lst_kpibyclass.append(_df_input[['cell'] + cols_feat][_df_input.class1 == j].groupby(['cell'])[cols_feat].max())        

        elif statsopt == 'median':
            for i, j in enumerate(pred_label):
                lst_kpibyclass.append(_df_input[['cell'] + cols_feat][_df_input.class1 == j].groupby(['cell'])[cols_feat].median())

        elif statsopt == 'average':
            for i, j in enumerate(pred_label):
                lst_kpibyclass.append(_df_input[['cell'] + cols_feat][_df_input.class1 == j].groupby(['cell'])[cols_feat].mean()) 
                

        for i, j in enumerate(pred_label):
            lst_tmp = lst_kpibyclass[i]
            lst_classbykpi = []
            for p, q in enumerate(cols_feat):
                    lst_classbykpi.append(lst_tmp.loc[:, q])
            mgcBoxkpi(lst_classbykpi, j, cols_feat)
        return self
  
    def mgcClassbyKpiMat(self, **kwargs):
        """
        return boxplot of each class
        """
        def_vals = {'pred_label' : self.pred_label,
                    'df_input' : self.df_kpi_cell_pred,
                    'cols_feat' : self.kpi_list, 
                    'statsopt' : 'max_min', 
                    'showflierx' : False}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        pred_label = kwargs['pred_label']
        _df_input = kwargs['df_input']
        cols_feat = kwargs['cols_feat']
        statsopt = kwargs['statsopt']
        _showflierx = kwargs['showflierx']

        # kpi columns to numeric
        _df_input[cols_feat] = _df_input[cols_feat].apply(pd.to_numeric, errors='coerce')
        
        lst_classbykpi = []


        kpi_list_invert = ['LTE_CQI_Avg', 'LTE_User_THP_DL', 'LTE_User_THP_UL', 
                           'LTE_RaCbraSuccRatio','LTE_Avg_PUSCH_SINR', 'LTE_Avg_PUCCH_SINR']
        
        kpi_list_invert = list(set(cols_feat).intersection(kpi_list_invert)).copy()
        
        # kpi columns to numeric
        #_df_input = df_raw.copy()
        _df_input[cols_feat] = _df_input[cols_feat].apply(pd.to_numeric, errors='coerce')

        # default whole cols_feat aggregate
        dict_togroup = dict(zip(cols_feat, ['max']*len(cols_feat)))
        
        # update default dictionary with specific kpi aggregrate by minimum
        dict_togroup.update(dict(zip(kpi_list_invert, ['min']*len(kpi_list_invert))))

        
        if statsopt == 'max_min':
            #df_pred_agg = _df_input.groupby(['cell', 'class1'])[cols_feat].max()
            df_pred_agg = _df_input[['cell', 'class1'] + cols_feat].groupby(['cell', 'class1']).agg(dict_togroup)
        elif statsopt == 'median':
            df_pred_agg = _df_input.groupby(['cell', 'class1'])[cols_feat].median()
        elif statsopt == 'average':
            df_pred_agg = _df_input.groupby(['cell', 'class1'])[cols_feat].mean()
            
        df_pred_agg.reset_index(inplace=True)
        
        #self.df_pred_agg = df_pred_agg
        
        for p, q in enumerate(cols_feat):
                lst_classbykpi.append(df_pred_agg.loc[:, ['class1', q]])        
                
                
        for i, j in enumerate(cols_feat):
            lst_tmp = lst_classbykpi[i]
            lst_plot = []
            for p, q in enumerate(pred_label):
                    lst_tmpx = lst_tmp[lst_tmp.class1 == q]
                    lst_plot.append(lst_tmpx.loc[:, j])
            mgcBoxclass(lst_plot, j, pred_label, showflierx = _showflierx)
        return self

    def mgcKdePlt(self, **kwargs):
        """
        return density line of each prediction category for KPIs
            Arguments:
                
        """    

        def_vals = {'df_input' : self.df_kpi_cell_pred,
                    'cols_feat' : self.kpi_list, 
                    'statsopt' : 'max_min'}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        _df_input = kwargs['df_input']
        cols_feat = kwargs['cols_feat']
        statsopt = kwargs['statsopt']
        
        params = {'df_raw': _df_input, 
                  'kpi_list': cols_feat, 
                  'statsopt': statsopt}
        _df_final = groupby_agg(**params) # 'max_min', 'median', 'average'

        mgcDensityPlt2(df_input = _df_final)
        
        
    def mgcTernaryScat(self, **kwargs):
        def_vals = {'pred_prob' : self.top3_pred}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        pred_prob = kwargs['pred_prob']
 
        mgcTernaryScat(pred_prob)
        return self
            
    def mgcTernaryHeat(self, **kwargs):
        def_vals = {'pred_prob' : self.top3_pred}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        pred_prob = kwargs['pred_prob']
 
        mgcTernaryHeat(pred_prob)
        return self
        
        
    def mgcHist2dMat(self, **kwargs):
        def_vals = {'pred_prob' : self.top3_pred}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        pred_prob = kwargs['pred_prob']
 
        mgcHist2d(pred_prob)
        return self