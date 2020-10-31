# import modules
import numpy as np
import pandas as pd

import random
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot 

import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import colorlover as cl
import sys
import re
from io import StringIO

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import matplotlib.pyplot as plt

text_source = "devkit==v1.0"

# to smooth learning curve
def smooth_curve(points, factor=0.8):
    """
    return smoothed curve on given data
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

                                                                                                                      
def linePlot(**kwargs):
    """
    return line plot
    
    Arguments:
        title: plot header name
        y_title: y-axis label name
        x: x-axis variables
        y1: first y-axis variable
        y2: second y-axis variable
    """

    def_vals = {"title" : 'title',
                "y_title" : 'y_title',
                "x" : 5,
                "y1" : random.sample(range(1, 100), 5),
                "y2" : random.sample(range(1, 100), 5)
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    title = kwargs['title']
    y_title = kwargs['y_title']
    x = kwargs['x']
    y1 = kwargs['y1']
    y2 = kwargs['y2']
    
    
    
    # String citing the data source

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

    x_title = 'Epoch' #.format(site1)  # x and y axis titles

    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
            size=13
        ),
        xaxis=go.XAxis(
            title=x_title,   # set x-axis title
            #range=xy_range,  # x-axis range
            zeroline=False   # remove x=0 line
        ),
        yaxis=go.YAxis(
            title=y_title,   # y-axis title
            #range=xy_range,  # y-axis range (same as x-axis)
            zeroline=False   # remove y=0 line
        ),
        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=False,  # remove legend
        autosize=False,    # custom size
        width=840,         # set figure width 
        height=620,         #  and height
        margin=dict(l=50,
                    r=30,
                    b=30,
                    t=50
        )
    )

    trace1 = go.Scatter(
        x=x,
        y=y1,
        mode='lines+markers',
        name = 'Train',
        marker=dict(
            size=2,
            color='red',            # set color to an array/list of desired values
            opacity=0.6
        )
    )

    trace2 = go.Scatter(
        x=x,
        y=y2,
        mode='lines+markers',
        name = 'Valid',
        marker=dict(
            size=2,
            color='blue',           # set color to an array/list of desired values
            opacity=0.6
        )
    )

    data = [trace1, trace2]

    fig = go.Figure(data=data, layout=layout)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})


###############################################
############ line plot multiple   #############
###############################################
def linePlot_acc_loss(**kwargs):
    """
    return multiple subplots in one plot
    Arguments:
        title: plot header name
        df_history: x, y variables in pandas dataframe
    """
 
    
    
    data = {'acc' : random.sample(range(1, 100), 5),
            'val_acc' : random.sample(range(1, 100), 5),
            'loss' : random.sample(range(1, 100), 5),
            'val_loss' : random.sample(range(1, 100), 5)
           }
    
    
    def_vals = {"title" : 'Learning History <br> Train, Validation',
                "df_history" : pd.DataFrame(data),
                "col_data1" : None,
                "col_data2" : None
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    title = kwargs['title']
    df_history = kwargs['df_history']
    col_data1 = kwargs['col_data1']
    col_data2 = kwargs['col_data2']

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


    x_title = 'Epoch'#.format(site1)  # x and y axis titles

    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
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
        width=840,         # set figure width 
        height=620,         #  and height
        margin=dict(l=100,
                    r=100,
                    b=50,
                    t=100
        )
    )


    fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True)

    for col in col_data1:
        fig.append_trace({'x': df_history.index, 'y': df_history[col], 'type': 'scatter', 'name': col}, 1, 1)
    for col in col_data2:
        fig.append_trace({'x': df_history.index, 'y': df_history[col], 'type': 'scatter', 'name': col}, 2, 1)

    fig.update(layout = layout)

    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

###############################################
############ scatter plot         #############
###############################################
def precision_recall(**kwargs):
    
    """
    to plot the precision and recall scatter on given classifcation report
    
    Arguments:
        plot_title: the title name of the plot
        df_report: classification report in pandas dataframe format
        dict_label: map integer variable to classification label
        dict_color: map color to plot trace
        
    """

    # label dimension
    n_label = 10
    
    values = ['label_' + str(i).zfill(2) for i in range(0,n_label)]
    keys = range(n_label) 
    cols = cl.interp(cl.scales['12']['qual']['Paired'], 20)[:n_label]
    
    def_vals = {"plot_title" : 'Top 1',
                "df_report" : [],
                "dict_label" : dict(zip(keys, values)),
                "dict_color" : dict(zip(values, cols))
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v) # update keywords arguments if not provided by user

    plot_title = kwargs['plot_title']
    df_report = kwargs['df_report']
    dict_label = kwargs['dict_label']
    dict_factor = {v: k for k, v in dict_label.items()}
    dict_color = kwargs['dict_color']
    
    df_plot = pd.DataFrame([], columns = ['x', 'y', 'z', 'size', 'col'])
    df_plot['x'], df_plot['y'], df_plot['z'], df_plot['sizei'], df_plot['col'] = df_report.precision, df_report.recall, \
    df_report['f1-score'], df_report['support']/df_report['support'].max(), df_report['label']

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

    title = 'Classification<br>\
    Precision and Recall ' + plot_title  # plot's title

    x_min, x_max, y_min, y_max = 0, 1.2, 0, 1.2


    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
            size=13
        ),
        xaxis = dict(
            range = [x_min, x_max], autorange = False, title='Precision',
            showgrid=True, gridwidth=2),
        yaxis = dict(
            range = [y_min, y_max], autorange = False, title='Recall',
            showgrid=True, gridwidth=2),


        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=True,  # remove legend
        autosize=False,    # custom size
        width=800,         # set figure width 
        height=600,         #  and height
        margin=dict(l=50,
                    r=50,
                    b=50,
                    t=50
        )
    )



    traces = []


    for i in df_plot.col.unique():
        df_by_class = df_plot[df_plot['col'] == i] 
        col_map = pd.Series(dict_color)[pd.Series(i).map(dict_factor)]


        traces.append(go.Scatter(
            x = df_by_class.x,
            y = df_by_class.y,
            mode = 'markers',
            #text = df_plot.col,
            name=i,
            marker=dict(
                size=df_by_class.sizei*100,
                sizemode='diameter',
                sizemin=15,
                #sizeref=max(size),
                color=col_map,          # set color to an array/list of desired values
                opacity=0.7
        )
        ))



    data = traces
    figure = go.Figure(data=data, layout=layout)

    figure['layout'] = {'shapes': [
            # unfilled circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': -1,
                'y0': -1,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgba(50, 171, 96, 1)',
                     'dash': 'dashdot'
                },
            },
            # unfilled Rectangle
            {
                'type': 'rect',
                'x0': 0.6,
                'y0': 0.6,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgba(128, 0, 128, 1)',
                    'dash': 'dashdot'
                },
            }
    ]}

    figure['layout'].update(
        hovermode='closest',  
        showlegend=False,     
        autosize=True,       

    )
    figure['layout'].update(images= [dict(
                  source= "image/0016_Blue_horizon.svg",
                  xref= "paper",
                  yref= "paper", xanchor="left", yanchor="bottom",
                  x= 0,
                  y= 0,
                  sizex= 0.1,
                  sizey= 0.1,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")])
    figure['layout'].update(layout)
    return iplot(figure,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})


###############################################
############ confusion matrix     #############
###############################################
def confs_mat_plot(**kwargs):
    
    """
    to plot confusion matrix
    Arguments:
        title_plot: the title name of the plot
        cnf_matrix: confusiob matrix in numpy array formnat   
        dict_color: map color to plot trace
    
    """

    import plotly.figure_factory as ff

    # label dimension
    n_label = 10
    
    values = ['label_' + str(i).zfill(2) for i in range(0,n_label)]
    keys = range(n_label) 
            
    def_vals = {"title_plot" : 'Confusion Matrix<br> Top 1',
                "cnf_matrix" : np.empty([n_label, n_label]),
                "dict_label" : dict(zip(keys, values))
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    title_plot = kwargs['title_plot']
    cnf_matrix = kwargs['cnf_matrix']
    dict_label = kwargs['dict_label']

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

    title = title_plot  # plot's title

    x_title = 'Predicted'#.format(site1)  # x and y axis titles
    y_title = 'True'#.format(site2)

    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
            size=13
        ),
        xaxis=go.XAxis(
            title=x_title,   # set x-axis title
            #range=xy_range,  # x-axis range
            zeroline=False   # remove x=0 line
        ),
        yaxis=go.YAxis(
            title=y_title,   # y-axis title
            #range=xy_range,  # y-axis range (same as x-axis)
            zeroline=False   # remove y=0 line
        ),
        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=False,  # remove legend
        autosize=False,    # custom size
        width=650,         # set figure width 
        height=520,         #  and height
        margin=dict(l=150,
                    r=100,
                    b=150,
                    t=100
        )

    )

    fig = ff.create_annotated_heatmap(
            x = list(dict_label.values()),
            y = list(dict_label.values()),
            z = cnf_matrix.round(decimals=2),
            opacity=0.9,
            colorbar=dict(
                title='Prediction Accuracy',
                titleside='right',
                titlefont=dict(
                    size=14,
                ),
            )
        )


    fig['layout'].update(width =800, height = 800, margin = dict(l=250,
                    r=50,
                    b=50,
                    t=300), title = title)

    fig['layout']['xaxis'].update(title = 'Predicted', tickangle = 90)
    fig['layout']['yaxis'].update(title = 'True')
    
    fig['layout'].update(images= [dict(
              source= "image/0016_Blue_horizon.svg",
              xref= "paper",
              yref= "paper", xanchor="left", yanchor="bottom",
              x= 0.5,
              y= 0.5,
              sizex= 0.5,
              sizey= 0.5,
              sizing= "stretch",
              opacity= 0.5,
              layer= "above")])
    #fig['layout'].update(layout)
    iplot(fig,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    
###############################################
############ density histogram    #############
###############################################
def dens_hist_plot(**kwargs):
    """
    plot prediction probability density histogram
    
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
    
    x = df['top1prob']
    y = df['top2prob']    

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
            family='PT Sans Narrow',  # global font
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
    trace2 = go.Histogram2dContour(
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
    #plot(fig, filename='network_predic.html', show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

###############################################
############ ROC AUC plot         #############
###############################################
def roc_plot(**kwargs):
    """
    plot AUC ROC in a single plot
    
    Arguments:
        y_test: ground truth of test dataset
        y_score: prediction probability based on classifier model #y_score = y_pred_prob
        nb_classes: number of class
        dict_label: map integer variable to classification label 
        dict_color: map color to plot trace
        
    """
   
    from sklearn.metrics import roc_curve, auc
    
    # label dimension
    n_label = 10
    
    values = ['label_' + str(i).zfill(2) for i in range(0, n_label)]
    keys = range(n_label) 
    cols = cl.interp(cl.scales['12']['qual']['Paired'], 20)[:n_label] 
    
    def_vals = {'y_test' : None,
                'y_score' : None,
                'nb_classes' : n_label,
                "dict_label" : dict(zip(keys, values)),
                "dict_color" : dict(zip(values, cols))
               }


    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

   
    y_test = kwargs['y_test']
    y_score = kwargs['y_score']
    nb_classes = kwargs['nb_classes']
    dict_label = kwargs['dict_label']
    dict_color = kwargs['dict_color']


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # String citing the data source

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

    title = 'Receiver operating characteristic <br>\
    x'  # plot's title

    x_min, x_max, y_min, y_max = 0, 1.0, 0, 1.0


    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
            size=13
        ),
        xaxis = dict(
                range = [x_min, x_max], autorange = False, title='False Positive Rate'),
        yaxis = dict(
                range = [y_min, y_max], autorange = False, title='True Positive Rate'),


        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=True,  # remove legend
        autosize=False,    # custom size
        width=800,         # set figure width 
        height=600,         #  and height
        margin=dict(l=100,
                    r=300,
                    b=100,
                    t=100
        )
    )



    # Create a trace
    traces = []

    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc = auc(fpr[i], tpr[i])
        col_map = pd.Series(i).map(dict_color)[0]
        # if np.isnan(col_map).sum() > 0:
        traces.append(go.Scatter(
            x=fpr[i],
            y=tpr[i],
            opacity=0.7,
            mode = 'lines',
            name=pd.Series(i).map(dict_label)[0],
            line=dict(
                color=col_map
            )
        ))
    data = traces

    figure = go.Figure(data=data, layout=layout)

    figure['layout'] = {'shapes': [
            # unfilled circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': 0,
                'y0': -1,
                'x1': 2,
                'y1': 1,
                'line': {
                    'color': 'rgba(50, 171, 96, 1)',
                    'dash': 'dashdot'
                },
            },
            # unfilled Rectangle
            {
                'type': 'line',
                'x0': 0,
                'y0': 0,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgb(128, 0, 128)',
                    'width': 4,
                    'dash': 'dot',
                },
            }
    ]}

    figure['layout'].update(
        hovermode='closest',  
        showlegend=False,     
        autosize=True,       

    )
    #figure['layout'].update(layout)
    
    figure['layout'].update(images= [dict(
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
    figure['layout'].update(layout)

    iplot(figure,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})
    

###############################################
############ Hinton Plot        #############
###############################################
def hinton_diag(**kwargs):
    
    """
    to plot the precision and recall scatter on given classifcation report
    
    Arguments:
        plot_title: the title name of the plot
        df_report: classification report in pandas dataframe format
        dict_label: map integer variable to classification label
        dict_color: map color to plot trace
        
    """

    # label dimension
    n_label = 10
    
    values = ['label_' + str(i).zfill(2) for i in range(0,n_label)]
    keys = range(n_label) 
    
    def_vals = {"title_plot" : 'Top 1',
                #"df" : [],
                "cm" : None,
                "plot_metric" : 'bycount',
                "dict_label" : dict(zip(keys, values)),
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v) # update keywords arguments if not provided by user

    title_plot = kwargs['title_plot']
    #df = kwargs['df']
    cm = kwargs['cm']
    plot_metric = kwargs['plot_metric']
    dict_label = kwargs['dict_label']
    dict_factor = {v: k for k, v in dict_label.items()}
    
    #cols = cl.interp(cl.scales['7']['qual']['Pastel1'], 16)
    cols = cl.interp(cl.scales['7']['qual']['Set2'], 16)
    colsx = [list(a) for a in zip([x / 10 for x in list(range(11))], cols)]

    
    # metrictype
    if plot_metric == 'bycount':
        df = pd.DataFrame(cm).stack().reset_index()
        df.columns = ['true', 'pred', 'count']
    
    elif plot_metric == 'precision':
        df = pd.DataFrame(cm/cm.sum(axis = 0)[np.newaxis, :]).stack().reset_index()
        #df = pd.DataFrame(cm).stack().reset_index()
        df.columns = ['true', 'pred', 'count']   

    elif plot_metric == 'recall':
        df = pd.DataFrame(cm/cm.sum(axis = 1)[:, np.newaxis]).stack().reset_index()
        #df = pd.DataFrame(cm).stack().reset_index()
        df.columns = ['true', 'pred', 'count']   
        
    def to_texts(df):
        textx = []
        if df['count'].dtypes == np.int32:
            textx = ["{0:.0f}".format(i) for i in df['count']]
        elif df['count'].dtypes == np.int64:
            textx = ["{0:.0f}".format(i) for i in df['count']]            
        elif df['count'].dtypes == np.float64:
            textx = ["{0:.2f}".format(i) for i in df['count']]
        return textx
        
    df_plot = pd.DataFrame([], columns = ['x', 'y', 'sizei', 'true_lb', 'pred_lb'])
    df_plot['x'], df_plot['y'], df_plot['sizei'], df_plot['true_lb'], df_plot['pred_lb'] = df['true'], df['pred'], \
    df['count'], df['true'].map(dict_factor), df['pred'].map(dict_factor)
        
    ratio_c = 500 / int(round(len(dict_label)/5.0)*5.0) 

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

    title = 'Classification<br>\
    Actual vs. Predict ' + title_plot  # plot's title


    # Make a layout object
    layout = go.Layout(
        title=title,  # set plot's title
        font=dict(
            family='PT Sans Narrow',  # global font
            size=13
        ),
        xaxis = dict(autorange = True, 
                     title='Actual',
                     showgrid=False,
                     showline=False,
                     zeroline=False,
                     gridwidth=1,
                     tickangle = 90,
                     ticks="", 
                     showticklabels=True,
                     mirror=True,
                     #linewidth=2,
                     tickvals=[i for i in range(len(dict_label))],
                     ticktext=list(dict_label.values())),
        yaxis = dict(autorange = True, 
                     title='Predict',
                     showgrid=False,
                     showline=False,
                     zeroline=False,                     
                     gridwidth=1,
                     ticks="", 
                     showticklabels=True,
                     mirror=True,
                     #linewidth=2,
                     tickvals=[i for i in range(len(dict_label))],
                     ticktext=list(dict_label.values())),


        annotations=go.Annotations([  # add annotation citing the data source
            make_anno()
        ]),
        showlegend=False,  # remove legend
        #autosize=True,    # custom size
        width=800,         # set figure width 
        height=800,         #  and height
        margin=dict(l=200,
                    r=50,
                    b=200,
                    t=100
        ),
        paper_bgcolor='rgba(245, 246, 249, 1)',
        plot_bgcolor='rgba(245, 246, 249, 1)',
    )


    trace = go.Scatter(
        x = df_plot['x'],
        y = df_plot['y'],
        mode = 'markers+text',
        text = pd.Series(to_texts(df)).replace({'0':np.nan, '0.00':np.nan}),
        hoverinfo = 'text',
        marker = dict(
            #color = '#FFBAD2',
            color = df_plot['sizei'],
            line = dict(width = 1),
            size = df_plot['sizei'],
            symbol = "square",
            sizeref = max(df_plot['sizei'])/ratio_c,
            sizemin = max(max(df_plot['sizei'])/ratio_c,10),
            #sizemin = 10,
            colorscale = colsx
        )
    )
    data = [trace]

    figure = go.Figure(data=data, layout=layout)

    return iplot(figure,  show_link=False, config={'modeBarButtonsToRemove': ['sendDataToCloud'], 'showLink': False, 'displaylogo' : False})

    
###############################################
############ top 3 prediction     #############
###############################################    
def top3(
        y_true=None,
        nb_classes=None,
        df_y_pred=None,
        dict_label=None):

    if len(dict_label) < 4:
        nlargest = len(dict_label)

        # reverse one-hot encoding, and categorical map to string label
        df_y_true = pd.DataFrame(
            pd.Series(y_true).map(dict_label), columns=['label'])

        # numeric to column name mapping
        col_name = (pd.Series(np.arange(nb_classes).tolist()
                              ).map(dict_label)).tolist()

        # assign columns names
        df_y_pred.columns = col_name

        arr_y_pred = df_y_pred.values  # transform to numpy array y_pred_probdf_y_pred

        order = np.argsort(-arr_y_pred, axis=1)[:, :nlargest]
        result = pd.DataFrame(df_y_pred.columns[order],
                              columns=['top{}'.format(i)
                                       for i in range(1, nlargest+1)],
                              index=df_y_pred.index)

        # retrieve top n columns values
        # Find sorted indices for each row
        sorted_row_idx = np.argsort(arr_y_pred, axis=1)[
            :, arr_y_pred.shape[1]-nlargest::]

        # Setup column indexing array
        col_idx = np.arange(arr_y_pred.shape[0])[:, None]

        # Use the column-row indices to get specific elements from input array.
        # Please note that since the column indexing array isn't of the same shape
        # as the sorted row indices, it will be broadcasted

        out = arr_y_pred[col_idx, sorted_row_idx]

        # top 3 and probability dataframe
        top_n_col_value = pd.DataFrame(
            out[:, ::-1], columns=['top{}'.format(i) for i in range(1, nlargest+1)])
        top_n_col_name_value = pd.concat([result, top_n_col_value], axis=1)
        top_n_col_name_value.columns = [
            'top1cla', 'top2cla', 'top1prob', 'top2prob']

        # for top 2 and top 3 classification report
        top_n_class = pd.concat([df_y_true, result], axis=1)
        top_n_class['pred2'] = np.where((top_n_class['top1'] == top_n_class['label']) | (top_n_class['top2'] == top_n_class['label']),
                                        top_n_class['label'], top_n_class['top1'])

    else:
        nlargest = 4
        # reverse one-hot encoding, and categorical map to string label
        df_y_true = pd.DataFrame(
            pd.Series(y_true).map(dict_label), columns=['label'])

        # numeric to column name mapping
        col_name = (pd.Series(np.arange(nb_classes).tolist()
                              ).map(dict_label)).tolist()

        # assign columns names
        df_y_pred.columns = col_name

        arr_y_pred = df_y_pred.values  # transform to numpy array y_pred_probdf_y_pred

        order = np.argsort(-arr_y_pred, axis=1)[:, :nlargest]
        result = pd.DataFrame(np.array(df_y_pred.columns)[order],
                              columns=['top{}'.format(i)
                                       for i in range(1, nlargest+1)],
                              index=df_y_pred.index)

        # retrieve top n columns values
        # Find sorted indices for each row
        sorted_row_idx = np.argsort(arr_y_pred, axis=1)[
            :, arr_y_pred.shape[1]-nlargest::]

        # Setup column indexing array
        col_idx = np.arange(arr_y_pred.shape[0])[:, None]

        # Use the column-row indices to get specific elements from input array.
        # Please note that since the column indexing array isn't of the same shape
        # as the sorted row indices, it will be broadcasted

        out = arr_y_pred[col_idx, sorted_row_idx]

        # top 3 and probability dataframe
        top_n_col_value = pd.DataFrame(
            out[:, ::-1], columns=['top{}'.format(i) for i in range(1, nlargest+1)])
        top_n_col_name_value = pd.concat([result, top_n_col_value], axis=1)
        top_n_col_name_value.columns = ['top1cla', 'top2cla', 'top3cla', 'top4cla', 'top1prob', 'top2prob', 'top3prob',
                                        'top4prob']

        # for top 2 and top 3 classification report
        top_n_class = pd.concat([df_y_true, result], axis=1)
        top_n_class['pred2'] = np.where((top_n_class['top1'] == top_n_class['label']) | (top_n_class['top2'] == top_n_class['label']),
                                        top_n_class['label'], top_n_class['top1'])
        top_n_class['pred3'] = np.where((top_n_class['top1'] == top_n_class['label']) | (top_n_class['top2'] == top_n_class['label'])
                                        | (top_n_class['top3'] == top_n_class['label']), top_n_class['label'], top_n_class['top1'])

    return arr_y_pred, top_n_class, top_n_col_name_value

###############################################
##### classification report to dataframe  #####
###############################################
def report_to_df(report):

    """
    function to convert classification report to dataframe (for visualisation plot)
    """

    #report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report = re.sub(r" +", " ", report).replace("micro avg", "micro_avg").replace("macro avg", "macro_avg").replace("weighted avg", "weighted_avg").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

###############################################
############ test to dataframe    #############
###############################################
def txt_to_df(top_n_class = None, dict_factor = None, nb_classes = None, dict_label = None):

    class_rpttop1 = classification_report(top_n_class['label'], top_n_class['top1'])
    class_report_top1 = classification_report(pd.Series(top_n_class['label']).map(dict_factor),
                                              pd.Series(top_n_class['top1']).map(dict_factor))
    df_report = report_to_df(class_report_top1)

    df_report = df_report.iloc[:nb_classes, :].copy()
    df_report.index = df_report.index.astype(int) 
    class_rpt_top1 = pd.concat([pd.DataFrame(pd.Series(df_report.index.tolist()).map(dict_label), columns = ['label']),
                                df_report], axis = 1)

    if len(dict_label) < 4:

        #txt report to df
        class_rpttop2 = classification_report(top_n_class['label'], top_n_class['pred2'])
        class_report_top2 = classification_report(pd.Series(top_n_class['label']).map(dict_factor), 
                                                  pd.Series(top_n_class['pred2']).map(dict_factor))
        df_report = report_to_df(class_report_top2)

        df_report = df_report.iloc[:nb_classes, :].copy()
        df_report.index = df_report.index.astype(int) 
        class_rpt_top2 = pd.concat([pd.DataFrame(pd.Series(df_report.index.tolist()).map(dict_label), columns = ['label']),
                                    df_report], axis = 1)

        to_return = [class_rpttop1, class_rpttop2, class_rpt_top1, class_rpt_top2]
                                    
    else:
    
        #txt report to df
        class_rpttop2 = classification_report(top_n_class['label'], top_n_class['pred2'])
        class_report_top2 = classification_report(pd.Series(top_n_class['label']).map(dict_factor), 
                                                  pd.Series(top_n_class['pred2']).map(dict_factor))
        df_report = report_to_df(class_report_top2)

        df_report = df_report.iloc[:nb_classes, :].copy()
        df_report.index = df_report.index.astype(int) 
        class_rpt_top2 = pd.concat([pd.DataFrame(pd.Series(df_report.index.tolist()).map(dict_label), columns = ['label']),
                                    df_report], axis = 1)

        #txt report to df
        class_rpttop3 = classification_report(top_n_class['label'], top_n_class['pred3'])
        class_report_top3 = classification_report(pd.Series(top_n_class['label']).map(dict_factor),
                                                  pd.Series(top_n_class['pred3']).map(dict_factor))
        df_report = report_to_df(class_report_top3)

        df_report = df_report.iloc[:nb_classes, :].copy()
        df_report.index = df_report.index.astype(int) 
        class_rpt_top3 = pd.concat([pd.DataFrame(pd.Series(df_report.index.tolist()).map(dict_label), columns = ['label']),
                                    df_report], axis = 1)   
        
        to_return = [class_rpttop1, class_rpttop2, class_rpttop3, class_rpt_top1, class_rpt_top2, class_rpt_top3]
    #return class_rpttop1, class_rpttop2, class_rpttop3, class_rpt_top1, class_rpt_top2, class_rpt_top3
    return to_return                    


###############################################
############ hinton matplotlib    #############
###############################################

def fto_texts(ls_num):
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

def _add_centered_square(ax, xy, area, lablz, text_size, **kwargs):
    """
    create hinton diagram element square with variable size according to weight matrix element value
    """
    size = np.sqrt(area)

    textz = fto_texts(lablz)
    loc = np.asarray(xy) - size/2.

    rect = mpatches.Rectangle(loc, size, size, **kwargs)

    label = ax.annotate(textz, xy=loc + size/2., fontsize=text_size, ha='center', va='center')

    ax.add_patch(rect)

def _ftext_size(area, max_weight, plot_metric = None):
    """
    custom text size accroding to weight matrix element value
    """
    
    plot_metric = plot_metric if plot_metric is not None else "precision"
    
    min_thresh = max_weight/6

    _text_size = {area > 0 and area < min_thresh: 8, 
                  area >= min_thresh and area < 2*min_thresh: 10,
                  area >= 2*min_thresh: 14}.get(True, 0)

    return _text_size

def _cm_metrics(cm, plot_metric = None):
    """
    convert basic confusion matrix to precsion or recall
    """
    plot_metric = plot_metric if plot_metric is not None else "precision"
    
    if plot_metric == 'bycount':
        cnf_matrix = cm
    
    elif plot_metric == 'recall':
        cnf_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]           
        ixs_isnan = np.isnan(cnf_matrix)
        cnf_matrix[ixs_isnan] = 0.0
    elif plot_metric == 'precision':
        cnf_matrix = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        ixs_isnan = np.isnan(cnf_matrix)
        cnf_matrix[ixs_isnan] = 0.0
    return cnf_matrix

def _cm_color(val, max_weight, plot_metric=None):
    """
    weight matrix element color variant dependent on confusion matrix plot option - bycount, precision or recall
    """
    plot_metric = plot_metric if plot_metric is not None else "precision"

    min_thresh = max_weight/6
    
    color = {val > 0 and val < min_thresh: (1, 0, 0, 0.5),
             val >= min_thresh and val < 2*min_thresh: (1, 1, 0, 0.5),
             val >= 2*min_thresh : (0, 1, 0, 0.5)}.get(True, (1, 1, 1, 0.0))

    return color


def hinton_mat(**kwargs):
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

    # label dimension
    n_label = 10
    
    values = ['label_' + str(i).zfill(2) for i in range(0,n_label)]
    keys = range(n_label) 
    
    def_vals = {"title_plot" : 'Top 1', 
                "max_weight": None, 
                "matrix" : None,
                "plot_metric" : 'bycount',
                "dict_label" : dict(zip(keys, values)), 
                "show_lgnd": False, 
                "ax": None
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
    
    dict_factor = {v: k for k, v in dict_label.items()} 

    # Plot confusion metrics with Hinton method

    
    """
    Draw Hinton diagram for visualizing a weight matrix.
    """
    fig, ax = plt.subplots(figsize = (10,10))
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
    matrix = _cm_metrics(matrix, plot_metric = plot_metric)
    if not max_weight:
         max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
    
    # for weight matrix element    
    for xy, val in np.ndenumerate(matrix):

        color = _cm_color(val, max_weight, plot_metric = plot_metric)
        text_size = _ftext_size(val, max_weight, plot_metric = plot_metric)

        areaz = max(0.3, np.abs(val)/max_weight)

        _add_centered_square(
            ax, np.asarray(xy), areaz, np.abs(val),
            text_size = text_size,
            color=color)

    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    plt.title("Confusion Matrix - {0}".format(title_plot), color='red', fontsize=18)
    plt.show()

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
    

###############################################
##### scatter of precision and recall   #######
###############################################
def scatPrecRec(df, max_weight=None, fill_col = True, marker_min = None, title_plt =  None):
    #import matplotlib.pyplot as plt
    """
    Return precision and recall scatter plot
    
    Arguments:
        df: classification report in datafrom format
        max_weight: maximum value that determine the size (in term of area) of the highest support value
        fill_col: boolean to show fill color of the marker or non-fill
        marker_min: minimum marker size
        title_plt: plot main title
    """

    feats = np.array(df[['precision', 'recall', 'support']])
    target = df['label']
    
    num_col = len(target)
    colors=plt.cm.tab20(np.linspace(0, 1, num_col))
    colors = tuple(map(tuple, colors))

    _result = [None]*(len(colors))
    _result[::2] = colors[::2]
    _result[1::2] = ['None'] * (num_col//2)

    if fill_col:
        result = colors
        edge_col = ['white'] * num_col
        line_styl = '-'
    else:
        result = _result
        edge_col = colors
        line_styl = '--'
    if not max_weight:
         max_weight = (2 ** np.ceil(np.log(np.abs(feats[:, 2]).sum()) / np.log(2)))/ 100
    #print(max_weight)
    #print(feats[:, 2].sum())
            
    f, ax = plt.subplots(figsize=(8, 8))

    # title and label
    plt.grid(linestyle = 'dashed', color = 'lightgrey', alpha = 0.5)
    plt.title('Recall vs. Precision {0}'.format(title_plt), fontsize = 20)
    plt.ylabel('Precision', fontsize = 16)
    plt.xlabel('Recall', fontsize = 16)

    # axis range
    ax.set_xlim([0,1.1])
    ax.set_ylim([0,1.1])

    # draw shhape
    square1 = plt.Rectangle((0.6, 0.6), 0.4, 0.4, clip_on = True,
                            linestyle = '-.', facecolor = 'none', alpha = 0.5,
                            edgecolor='purple', linewidth = 2)
    circle1 = plt.Circle((0, 0), 1, clip_on = True,
                         linestyle = '-.', facecolor = 'none', alpha = 0.5,
                         edgecolor='green')
    ax.add_artist(square1)
    ax.add_artist(circle1)
    ax.patch.set_facecolor('None')
    ax.patch.set_alpha(0.0)
    cols_face = []
    if not marker_min:
        marker_min  = 20
    # draw points
    for i, j in enumerate(np.unique(target)):
        mask = target == j
        plt.scatter(feats[mask, 0], feats[mask, 1], label=j, 
                    facecolor = result[i], edgecolor=edge_col[i],
                    s = np.where((feats[mask, 2]*max_weight) < marker_min**2, marker_min**2, (feats[mask, 2]*max_weight)), 
                    linestyle = line_styl,
                    alpha = 0.7)

    # custom legend size
    dict_msizes = [100] * num_col
    markers = []
    for i, size in enumerate(dict_msizes):
        markers.append(plt.scatter([],[], 
                                   s=size, label=target[i], 
                                   marker='o', facecolors=result[i], 
                                   alpha = 0.9, edgecolor = edge_col[i], linestyle = line_styl))

    ax.legend(handles = markers,
                scatterpoints=1,
               loc='best',
               ncol=1, 
               markerscale=1.6, 
               numpoints=1,
               borderpad=1, 
               handlelength=3, 
               labelspacing=0.5,
               fontsize=14, frameon=False, 
               bbox_to_anchor=(1,1))

    plt.show()
    #plt.show(block=True)


###############################################
############ plot class           #############
###############################################

class Mgcplot():
    """
    plot machine learning model evaluation metrics
    
    Arguments:
        y_true <1d array>: the vector of true label
        y_pred_prob <2d array>: nsamples*nclasses
        nb_classes <int>: the number of classed
        dict_label <dict>: {id: the name of id}
        
    """
    

    def __init__(self, **kwargs):
        super().__init__()
      
        
        def_vals = {
                    "y_true" : None,
                    "y_pred_prob" : None,
                    "y2_true" : None,
                    "y2_pred_prob" : None,
                    "nb_classes" : 13,
                    "nb_classes2" : 4,
                    "dict_label" : None,
                    "dict_label2" : None,
                    "input_branch" : 'single'} # default parameters value

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        self.y_true = kwargs['y_true']
        self.y_pred_prob = kwargs['y_pred_prob']
        self.y2_true = kwargs['y2_true']
        self.y2_pred_prob = kwargs['y2_pred_prob']
        self.nb_classes = kwargs['nb_classes']
        self.nb_classes2 = kwargs['nb_classes2']
        self.dict_label = kwargs['dict_label']
        self.dict_label2 = kwargs['dict_label2']                
        self.input_branch = kwargs['input_branch']
        
        self.y_pred = np.argmax(self.y_pred_prob, axis=1)
        
        # default label dictionary if users do not provide
        values = ['label_' + str(i).zfill(2) for i in range(0,self.nb_classes)]
        keys = range(self.nb_classes)
        
     
            
        if self.dict_label is None:
            self.dict_label = dict(zip(keys, values))
        else:
            self.dict_label = kwargs['dict_label']
        
        self.dict_factor = {v: k for k, v in self.dict_label.items()}
        
        values2 = ['label_' + str(i).zfill(2) for i in range(0,self.nb_classes2)]
        keys2 = range(self.nb_classes2)
        
        if self.dict_label2 is None:
            self.dict_label2 = dict(zip(keys2, values2))
        else:
            self.dict_label2 = kwargs['dict_label2']
        
        self.dict_factor2 = {v: k for k, v in self.dict_label2.items()}
            
        if self.input_branch == "dual":
            self.y2_pred = np.argmax(self.y2_pred_prob, axis=1)

            _y_true = [self.y_true, self.y2_true]
            _y_pred = [self.y_pred, self.y2_pred]
            _y_pred_prob = [self.y_pred_prob, self.y2_pred_prob]
            self.y_pred_list_df = []
            for i in range(2):
                self.y_pred_list_df.append(pd.DataFrame(self._y_pred_prob[i]))

            nb_classes_c = [self.nb_classes, self.nb_classes2]
            self.dict_label_c = [self.dict_label, self.dict_label2]

            dict_factor_c = [self.dict_factor, self.dict_factor2]

        elif self.input_branch == "single":
            _y_true = [self.y_true]
            _y_pred = [self.y_pred]
            self.y_pred_list_df = [pd.DataFrame(self.y_pred_prob)]

            nb_classes_c = [self.nb_classes]
            self.dict_label_c = [self.dict_label]

            dict_factor_c = [self.dict_factor]
        ############################################################################################


        #nlargest = 4
        
        self.list_y_pred = []
        self.list_top_n_class = []
        self.list_top_n_col_name = []

        for i in range(len(_y_true)):
            arr_y_pred, top_n_class, top_n_col_name_value = top3(
                y_true = _y_true[i],
                nb_classes = nb_classes_c[i], 
                df_y_pred = self.y_pred_list_df[i], 
                dict_label = self.dict_label_c[i])

            self.list_y_pred.append(arr_y_pred)
            self.list_top_n_class.append(top_n_class)
            self.list_top_n_col_name.append(top_n_col_name_value)  
        
        self.class_rpttop1 = []
        self.class_rpttop2 = []
        self.class_rpttop3 = []
        self.class_rpt_top1 = []
        self.class_rpt_top2 = []
        self.class_rpt_top3 = []            
        
        if self.y_true is not None:
            for i in range(len(self.dict_label_c)):
                if len(self.dict_label_c[i]) < 4:
                    class_rpttop1, class_rpttop2, class_rpt_top1, class_rpt_top2 = txt_to_df(self.list_top_n_class[i],
                                                                    
                                            dict_factor_c[i],
                                            nb_classes_c[i],
                                            self.dict_label_c[i])
                    
                    self.class_rpttop1.append(class_rpttop1)
                    self.class_rpttop2.append(class_rpttop2)
                    self.class_rpt_top1.append(class_rpt_top1)
                    self.class_rpt_top2.append(class_rpt_top2)
                
                else:
                    class_rpttop1, class_rpttop2, class_rpttop3, class_rpt_top1, class_rpt_top2, \
                    class_rpt_top3 = txt_to_df(self.list_top_n_class[i],
                                            dict_factor_c[i],
                                            nb_classes_c[i],
                                            self.dict_label_c[i])
                    
                    self.class_rpttop1.append(class_rpttop1)
                    self.class_rpttop2.append(class_rpttop2)
                    self.class_rpttop3.append(class_rpttop3)
                    self.class_rpt_top1.append(class_rpt_top1)
                    self.class_rpt_top2.append(class_rpt_top2)
                    self.class_rpt_top3.append(class_rpt_top3)

        self.mgcConfMat()
        # self.mgcScatterMat(max_weight = 5, fill_col = True, marker_min = 10)

##########################################################
############  Classification Report ######################
##########################################################

    def mgc_plot_metrics(self):

        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('retina')
        
        self.mgcHintonTop1()

        self.mgcHintonTop2()

        self.mgcHintonTop3()

        from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot

        init_notebook_mode()

        print('Density Histogram')
        print("+++++++++++++++++++++++++++")
        self.mgcDensHist()

        print("ROC AUC")
        self.mgcAuc()

##########################################################
############  Classification Report ######################
##########################################################

    def mgcClassReport(self, **kwargs):
        self.df_class_rpttop1 = []
        self.df_class_rpttop2 = []
        self.df_class_rpttop3 = []
        for i in range(len(self.dict_label_c)):
            if len(self.dict_label_c[i]) < 4:
                self.df_class_rpttop1.append(report_to_df(self.class_rpttop1[i]))
                self.df_class_rpttop2.append(report_to_df(self.class_rpttop2[i]))
            
            else:
                self.df_class_rpttop1.append(report_to_df(self.class_rpttop1[i]))
                self.df_class_rpttop2.append(report_to_df(self.class_rpttop2[i]))
                self.df_class_rpttop3.append(report_to_df(self.class_rpttop3[i]))
        return self
      

##########################################################
############  plot single line      ######################
##########################################################

    def mgcLine(self, **kwargs):
        """
        return model training history single plot
        Arguments:
            df_history: training history in pandas dataframe
            
        """
        if self.input_branch == 'dual':
            df_history = kwargs['df_history']
            _col_data1 = ['dense_3_acc', 'dense_4_acc', 'val_dense_3_acc', 'val_dense_4_acc']
            _col_data2 = ['dense_3_loss', 'dense_4_loss', 'val_dense_3_loss', 'val_dense_4_loss']
        elif self.input_branch == 'single':
            df_history = kwargs['df_history']
            _col_data1 = ['acc', 'val_acc']
            _col_data2 = ['loss', 'val_loss'] 
            
        def_vals = {"df_history" : [],
                    "col_data1" : _col_data1,
                    "col_data2" : _col_data2
                   }
        
        
        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        col_data1 = kwargs['col_data1']
        col_data2 = kwargs['col_data2']
        
        self.plotAccLoss = linePlot_acc_loss(df_history = df_history, col_data1 = col_data1, col_data2 = col_data2)
        
        return self

        
##########################################################
############  plot learning rate    ######################
##########################################################

    def mgcLrLine(self, **kwargs):
        """
        return model training history single plot
        Arguments:
            df_history: training history in pandas dataframe
            
        """
        if self.input_branch == 'dual':
            df_history = kwargs['df_history']
            _col_data1 = ['dense_3_acc', 'dense_4_acc', 'val_dense_3_acc', 'val_dense_4_acc']
            _col_data2 = ['lr']
        elif self.input_branch == 'single':
            df_history = kwargs['df_history']
            _col_data1 = ['loss', 'val_loss']
            _col_data2 = ['lr']
            
        def_vals = {"df_history" : [],
                    "col_data1" : _col_data1,
                    "col_data2" : _col_data2
                   }
        
        
        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        col_data1 = kwargs['col_data1']
        col_data2 = kwargs['col_data2']
        
        self.plotAccLoss = linePlot_acc_loss(df_history = df_history, col_data1 = col_data1, col_data2 = col_data2)
        
        return self        
        
        
##########################################################
############  plot multiple lines   ######################
##########################################################

    def mgcLineS(self, **kwargs):
        
        """
        plot model training history, multiple plots
        """
        
        self.plotLine = linePlot(**kwargs)       
        return self
    
##########################################################
############  plot scatter          ######################
##########################################################    
        
    def mgcScatter(self):
    
        """
        plot precision recall scatter for top 2 and top 3 prediction
        """               

        nb_classes_c = [self.nb_classes, self.nb_classes2]
        dict_label_c = [self.dict_label, self.dict_label2]
        cols = []
        dict_color = []

        for i in range(len(nb_classes_c)):
            cols.append(cl.interp(cl.scales['12']['qual']['Paired'], 20)[:nb_classes_c[i]])
            dict_color.append(dict(zip(dict_label_c[i].values(), cols[i])))

            def b1_top1():
                return precision_recall(plot_title = 'Top 1',
                                        df_report = self.class_rpt_top1[0], 
                                        dict_label = dict_label_c[0],
                                        dict_color = dict_color[0])

            def b1_top2():
                return precision_recall(plot_title = 'Top 2', 
                                        df_report = self.class_rpt_top2[0],
                                        dict_label = dict_label_c[0], 
                                        dict_color = dict_color[0])

            def b1_top3():
                return precision_recall(plot_title = 'Top 3', 
                                        df_report = self.class_rpt_top3[0], 
                                        dict_label = dict_label_c[0], 
                                        dict_color = dict_color[0])
            def b2_top1():
                return precision_recall(plot_title = 'Top 1',
                                        df_report = self.class_rpt_top1[1], 
                                        dict_label = dict_label_c[1],
                                        dict_color = dict_color[1])

            def b2_top2():
                return precision_recall(plot_title = 'Top 2', 
                                        df_report = self.class_rpt_top2[1],
                                        dict_label = dict_label_c[1], 
                                        dict_color = dict_color[1])

            def b2_top3():
                return precision_recall(plot_title = 'Top 3', 
                                        df_report = self.class_rpt_top3[1], 
                                        dict_label = dict_label_c[1], 
                                        dict_color = dict_color[1])
        if self.input_branch == 'dual':
            self.b1_top1 = b1_top1()
            self.b1_top2 = b1_top2()
            self.b1_top3 = b1_top3()
            self.b2_top1 = b2_top1()
            self.b2_top2 = b2_top2()
            self.b2_top3 = b2_top3()
        elif self.input_branch == 'single':
            self.b1_top1 = b1_top1()
            self.b1_top2 = b1_top2()
            self.b1_top3 = b1_top3()
            
        
        return self
        
##########################################################
#########  plot scatter with matplotlib      #############
##########################################################    
        
    def mgcScatterMat(self, max_weight = None, fill_col = True, marker_min = None):
    
        """
        plot precision recall scatter for top 2 and top 3 prediction
        """               
        if self.input_branch == "single":
            nb_classes_c = [self.nb_classes]
            dict_label_c = [self.dict_label]
            cols = []
            dict_color = []
        
        elif self.input_branch == "dual":
            nb_classes_c = [self.nb_classes, self.nb_classes2]
            dict_label_c = [self.dict_label, self.dict_label2]
            cols = []
            dict_color = []

        for i in range(len(nb_classes_c)):
            cols.append(cl.interp(cl.scales['12']['qual']['Paired'], 20)[:nb_classes_c[i]])
            dict_color.append(dict(zip(dict_label_c[i].values(), cols[i])))
            
            if len(dict_label_c[i]) < 4:
                def b1_top1():
                    return scatPrecRec(self.class_rpt_top1[0], max_weight = max_weight, marker_min = marker_min, fill_col = fill_col, title_plt = 'Top 1')

                def b1_top2():
                    return scatPrecRec(self.class_rpt_top2[0], max_weight = max_weight, marker_min = marker_min, fill_col = fill_col, title_plt = 'Top 2')


                def b2_top1():
                    return precision_recall(plot_title = 'Top 1',
                                            df_report = self.class_rpt_top1[1], 
                                            dict_label = dict_label_c[1],
                                            dict_color = dict_color[1])

                def b2_top2():
                    return precision_recall(plot_title = 'Top 2', 
                                            df_report = self.class_rpt_top2[1],
                                            dict_label = dict_label_c[1], 
                                            dict_color = dict_color[1])

                if self.input_branch == 'dual':
                    self.b1_top1 = b1_top1()
                    self.b1_top2 = b1_top2()

                    self.b2_top1 = b2_top1()
                    self.b2_top2 = b2_top2()

                elif self.input_branch == 'single':
                    self.b1_top1 = b1_top1()
                    self.b1_top2 = b1_top2()

                
            else:
                                                                                                                                                      

                def b1_top1():
                    return scatPrecRec(self.class_rpt_top1[0], max_weight = max_weight, marker_min = marker_min, fill_col = fill_col, title_plt = 'Top 1')
                                                                            
                                                                     
                                                                   

                def b1_top2():
                    return scatPrecRec(self.class_rpt_top2[0], max_weight = max_weight, marker_min = marker_min, fill_col = fill_col, title_plt = 'Top 2')
                                                                           
                                                                      
                                                                   

                    
                def b1_top3():
                    return scatPrecRec(self.class_rpt_top3[0], max_weight = max_weight, marker_min = marker_min, fill_col = fill_col, title_plt = 'Top 3')

                def b2_top1():
                    return precision_recall(plot_title = 'Top 1',
                                            df_report = self.class_rpt_top1[1], 
                                            dict_label = dict_label_c[1],
                                            dict_color = dict_color[1])

                def b2_top2():
                    return precision_recall(plot_title = 'Top 2', 
                                            df_report = self.class_rpt_top2[1],
                                            dict_label = dict_label_c[1], 
                                            dict_color = dict_color[1])

                def b2_top3():
                    return precision_recall(plot_title = 'Top 3', 
                                            df_report = self.class_rpt_top3[1], 
                                            dict_label = dict_label_c[1], 
                                            dict_color = dict_color[1])
                if self.input_branch == 'dual':
                    self.b1_top1 = b1_top1()
                    self.b1_top2 = b1_top2()
                    self.b1_top3 = b1_top3()
                    self.b2_top1 = b2_top1()
                    self.b2_top2 = b2_top2()
                    self.b2_top3 = b2_top3()
                elif self.input_branch == 'single':
                    self.b1_top1 = b1_top1()
                    self.b1_top2 = b1_top2()
                    self.b1_top3 = b1_top3()
            
        
        return self
        
    
##########################################################
############  create confusion matrix ####################
##########################################################    
    
    def mgcConfMat(self, **kwargs):
        """
        predict from y_test and generate confusion matrix 
        """
        
        
        def conf_sub(nb_classes = None, list_top_n_class = None, dict_factor = None):
            
            if len(dict_factor) < 4:
                          
                y_pred = pd.Series(list_top_n_class['top1']).map(dict_factor)
                #print(y_pred)
                y_pred2 = pd.Series(list_top_n_class['pred2']).map(dict_factor)
                #print(y_pred2)
                y_true = pd.Series(list_top_n_class['label']).map(dict_factor)
                
                # top1
                cnf_matrix = confusion_matrix(y_true, y_pred, labels = range(nb_classes))
                #print(cnf_matrix)
                #print("**********************************************")

                cnf_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
                #print(cnf_norm)
                #print("**********************************************")
                
                cnf_norm_prec = cnf_matrix.astype('float') / cnf_matrix.sum(axis=0)[np.newaxis, :]
                #print(cnf_norm_prec)
                #print("**********************************************")
                
                
                # top2
                cnf_matrix_top2 = confusion_matrix(y_true, y_pred2, labels = range(nb_classes))
                cnf_norm_top2 = cnf_matrix_top2.astype('float') / cnf_matrix_top2.sum(axis=1)[:, np.newaxis]
                cnf_norm_prec_top2 = cnf_matrix_top2.astype('float') / cnf_matrix_top2.sum(axis=0)[np.newaxis, :]
                
                
                return [cnf_matrix, cnf_norm.round(decimals=2), cnf_norm_prec.round(decimals=2), 
                        cnf_matrix_top2, cnf_norm_top2.round(decimals=2), cnf_norm_prec_top2.round(decimals=2)]
            
            else:
                                                                                           
                                                                                                        
                                                                                                              
            
                y_pred = pd.Series(list_top_n_class['top1']).map(dict_factor)
                #print(y_pred)
                y_pred2 = pd.Series(list_top_n_class['pred2']).map(dict_factor)
                #print(y_pred2)
                y_pred3 = pd.Series(list_top_n_class['pred3']).map(dict_factor)
                #print(y_pred3)
                y_true = pd.Series(list_top_n_class['label']).map(dict_factor)
                
                # top1
                cnf_matrix = confusion_matrix(y_true, y_pred, labels = range(nb_classes))
                #print(cnf_matrix)
                #print("**********************************************")

                cnf_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
                #print(cnf_norm)
                #print("**********************************************")
                
                cnf_norm_prec = cnf_matrix.astype('float') / cnf_matrix.sum(axis=0)[np.newaxis, :]
                #print(cnf_norm_prec)
                #print("**********************************************")
                ixs_isnan = np.isnan(cnf_norm_prec)
                cnf_norm_prec[ixs_isnan] = 0.0
                
                # top2
                cnf_matrix_top2 = confusion_matrix(y_true, y_pred2, labels = range(nb_classes))
                cnf_norm_top2 = cnf_matrix_top2.astype('float') / cnf_matrix_top2.sum(axis=1)[:, np.newaxis]
                cnf_norm_prec_top2 = cnf_matrix_top2.astype('float') / cnf_matrix_top2.sum(axis=0)[np.newaxis, :]
                ixs_isnan = np.isnan(cnf_norm_prec_top2)
                cnf_norm_prec_top2[ixs_isnan] = 0.0                
                # top3
                cnf_matrix_top3 = confusion_matrix(y_true, y_pred3, labels = range(nb_classes))
                cnf_norm_top3 = cnf_matrix_top3.astype('float') / cnf_matrix_top3.sum(axis=1)[:, np.newaxis]
                cnf_norm_prec_top3 = cnf_matrix_top3.astype('float') / cnf_matrix_top3.sum(axis=0)[np.newaxis, :] 
                ixs_isnan = np.isnan(cnf_norm_prec_top3)
                cnf_norm_prec_top3[ixs_isnan] = 0.0                
                return [cnf_matrix, cnf_norm.round(decimals=2), cnf_norm_prec.round(decimals=2), 
                        cnf_matrix_top2, cnf_norm_top2.round(decimals=2), cnf_norm_prec_top2.round(decimals=2), 
                        cnf_matrix_top3, cnf_norm_top3.round(decimals=2), cnf_norm_prec_top3.round(decimals=2)]
            
        
        y_true_c = [self.y_true, self.y2_true]
        nb_classes_c = [self.nb_classes, self.nb_classes2]
        dict_label_c = [self.dict_label, self.dict_label2]
        dict_factor_c = [self.dict_factor, self.dict_factor2]
        _list_top_n_class = self.list_top_n_class
        
        conf_mat_main = []
        #print(len(y_true_c))
        
        for k in range(len(self.dict_label_c)):
            #print(_list_top_n_class[k])
            
            _conf_tmp = conf_sub(nb_classes = nb_classes_c[k],
                                 list_top_n_class = _list_top_n_class[k],
                                 dict_factor = dict_factor_c[k])
            
            conf_mat_main.append(_conf_tmp)
        
        self.conf_mat_main = conf_mat_main
        
        return self
    
##########################################################
############  plot confusion matrix top 1 ################
##########################################################

    def plotConfMatTop1(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Recall", "Precision"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        
        for i in range(nb_iter):
            for j in range(3):

                params = {"title_plot" : plot_title[j],
                          "cnf_matrix" : conf_mat[i][j],
                          "dict_label": dict_label[i]}

                cm_plot = confs_mat_plot(**params)

        self.cm_plot = cm_plot
        return self
    
##########################################################
############  plot confusion matrix top 2 ################
##########################################################    

    def plotConfMatTop2(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Recall", "Precision"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        
        for i in range(nb_iter):
            for j in range(3, 6):

                params = {"title_plot" : plot_title[j],
                          "cnf_matrix" : conf_mat[i][j],
                          "dict_label": dict_label[i]}

                cm_plot = confs_mat_plot(**params)

        self.cm_plot = cm_plot
        return self    
    
##########################################################
############  plot confusion matrix top 3 ################
##########################################################    

    def plotConfMatTop3(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Recall", "Precision"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        
        for i in range(nb_iter):
            for j in range(6, 9):

                params = {"title_plot" : plot_title[j],
                          "cnf_matrix" : conf_mat[i][j],
                          "dict_label": dict_label[i]}

                cm_plot = confs_mat_plot(**params)

        self.cm_plot = cm_plot
        return self  
    
##########################################################
############  plot density histogram      ################
##########################################################    
    
    def mgcDensHist(self, **kwargs):
        """
        density histogram
        """
        from plotly.offline import init_notebook_mode
        init_notebook_mode()

        nb_iter = len(self.dict_label_c)
        densHist = []
        for i in range(nb_iter):
            
            densHist.append(dens_hist_plot(df = self.list_top_n_col_name[i].loc[:,['top1prob', 'top2prob']]))
        self.plotDensHist = densHist
        return self  
    
##########################################################
############  plot AUC                    ################
##########################################################      
    

    def mgcAuc(self, **kwargs):
        """
        return ROC AUC plot for all classes
        """
        #def plotAuc(nb_classes_c = None, dict_label_c = None, y_true_c = None, y_pred_list_df = None):
        
        nb_iter = len(self.dict_label_c)
        
        plotAucx = []
        nb_classes_c = [self.nb_classes, self.nb_classes2]
        y_true_c = [self.y_true, self.y2_true]
        y_pred_list_df_c = self.y_pred_list_df
        
        for i in range(nb_iter):
            enc = OneHotEncoder().fit(y_true_c[i].reshape(-1,1))
            y_true_onehot = enc.transform(y_true_c[i].reshape(-1,1)).toarray()
            cols = cl.interp(cl.scales['12']['qual']['Paired'], nb_classes_c[i])[:nb_classes_c[i]] # color map
            dict_colorx = dict(zip(self.dict_label_c[i].keys(), cols))
            _plotAuc_tmp = roc_plot(y_test = y_true_onehot, 
                            y_score = y_pred_list_df_c[i].values, 
                            nb_classes = nb_classes_c[i], 
                            dict_label = self.dict_label_c[i], 
                            dict_color = dict_colorx)

            plotAucx.append(_plotAuc_tmp)
        
        self.mgcAucx = plotAucx
        return self
    
##########################################################
############  plot hinton diagram  top 1  ################
##########################################################      
    
    def plotHintonTop1(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall']
        #_plot_metric = ['bycount', 'bycount', 'bycount']
        
        for i in range(nb_iter):
            for j in range(3):

                params = {"title_plot" : plot_title[j],
                          "cm" : conf_mat[i][0],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i]}

                cm_plot = hinton_diag(**params)

        self.cm_plot = cm_plot
        return self
    
##########################################################
############  plot hinton diagram  top 2  ################
##########################################################    
    
    def plotHintonTop2(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall', 'bycount', 'precision', 'recall', 'bycount', 'precision', 'recall']
        
        for i in range(nb_iter):
            for j in range(3, 6):

                params = {"title_plot" : plot_title[j],
                          "cm" : conf_mat[i][3],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i]}

                cm_plot = hinton_diag(**params)

        self.cm_plot = cm_plot
        return self
    
##########################################################
############  plot hinton diagram  top 3 ################
##########################################################    
       
    def plotHintonTop3(self, **kwargs):

        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)

        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall', 'bycount', 'precision', 'recall', 'bycount', 'precision', 'recall']

        for i in range(nb_iter):
            for j in range(6, 9):

                params = {"title_plot" : plot_title[j],
                          "cm" : conf_mat[i][6],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i]}

                cm_plot = hinton_diag(**params)

        self.cm_plot = cm_plot
        return self
        
        
        
    def mgcHintonTop1(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall']
        #_plot_metric = ['bycount', 'bycount', 'bycount']
        
        for i in range(nb_iter):
            for j in range(3):
                params = {"title_plot" : plot_title[j], 
                          "max_weight": None,
                          "matrix" : conf_mat[i][0],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i], 
                          "show_lgnd": False}

                plt.figure(figsize=(10, 10))
                hinton_mat(**params)
        return self
        

    def mgcHintonTop2(self, **kwargs):
        
        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)
        
        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall', 'bycount', 'precision', 'recall', 'bycount', 'precision', 'recall']
        
        for i in range(nb_iter):
            for j in range(3, 6):

                
                params = {"title_plot" : plot_title[j], 
                          "max_weight": None,
                          "matrix" : conf_mat[i][3],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i], 
                          "show_lgnd": False}

                hinton_mat(**params)
                
        return self
        
    def mgcHintonTop3(self, **kwargs):

        conf_mat = self.conf_mat_main
        dict_label = self.dict_label_c

        topx = ['Top 1', 'Top2', 'Top3']
        confx = ['Confusion Matrix', "Precision", "Recall"]
        plot_title = []
        for a in range(3):
            for b in range(3):
                _title_tmp = topx[a] + " " + confx[b]
                plot_title.append(_title_tmp)

        nb_iter = len(self.dict_label_c)
        _plot_metric = ['bycount', 'precision', 'recall', 'bycount', 'precision', 'recall', 'bycount', 'precision', 'recall']

        for i in range(nb_iter):
            for j in range(6, 9):

                params = {"title_plot" : plot_title[j], 
                          "max_weight": None,
                          "matrix" : conf_mat[i][6],
                          "plot_metric" : _plot_metric[j],
                          "dict_label": dict_label[i], 
                          "show_lgnd": False}

                hinton_mat(**params)
        return self
