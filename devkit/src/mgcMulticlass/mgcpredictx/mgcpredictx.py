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

__version__ = '0.0.15'

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
        this is local version
        
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
        # Load prediction model
        
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'best_modelz.hdf5')
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
        
    def mgcLoadNormData(self, **kwargs):

        #### set default keyword parameter value, else update with user input params
        def_vals = {"feat_norm_train": False}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
            
        self.kpi_quantile = kwargs['kpi_quantile']

        self.feat_norm = kwargs['feat_norm']
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
                         'LTE_PrbUtilDl_Avg': 100,
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
            
        # transform to wide table
        df_raw_norm_x_w = pd.DataFrame(self.df_raw.groupby(['cell', 'hour']).sum().unstack('hour'))
        
        # drop nan rows
        df_raw_norm_x_w.dropna(how='any', inplace = True)
        
        df_raw_norm_x_w.reset_index(inplace=True)
        # rename the columns and pattern the hierarchical header
        new_cols = [''.join(t) for t in df_raw_norm_x_w.columns]
        df_raw_norm_x_w.columns = new_cols

        df_raw_norm_x_w.reset_index(level = 0, inplace=True)
        
        df_raw_norm_x_w.index = df_raw_norm_x_w['cell']
        
        # rearrange columns by kpi and followed by hour
        new_list = []
        for i in range(24):
            new_list.extend([s + str(i).zfill(2) for s in self.kpi_list])
        
        """
        # rearrange columns by hour then kpi
        new_list = []
        for i in range(len(self.kpi_list)):
            new_list.extend(self.kpi_list[i] + str(s).zfill(2) for s in range(24))"""
        
        # rearrange col to kpi then  hour (vs. hour then kpi)
        df_raw_norm_x_w = df_raw_norm_x_w[new_list]
        self.df_raw_norm_x_w = df_raw_norm_x_w
        
        feat_dataset = np.array(df_raw_norm_x_w.iloc[:, :])
        self.feat_dataset = feat_dataset.astype('float32')

        # Check value range / whole number for current data set
        def kpiDesc():
                current_data = self.feat_dataset.reshape(self.feat_dataset.shape[0], self.nb_hours, self.nb_kpis)
                print('min-max normalized distribution')
                arr_x = current_data.reshape(-1, current_data.shape[2])
                pd.DataFrame(arr_x, columns = self.kpi_list).hist(figsize=(26, 16), normed = True)
                plt.show()

                
        def _feat_normalize(feat_dataset = None):
            # Normalize input data to be in the range of 0-1. 
            # Note: Thresholds are based on the data analysis
            # norm := x - min(x) / max(x) - min(x)
            feat_dataset_norm = np.copy(feat_dataset)
            feat_dataset_norm = feat_dataset_norm.reshape(feat_dataset_norm.shape[0], self.nb_hours, self.nb_kpis)
            
            if self.kpi_quantile:
                _norm_thresh = self.dict_kpi_quantile
            else:
                _norm_thresh = self.norm_thresh
            
            
            for i, j in enumerate(self.kpi_list):
                # Minimum value of all values corresponding to current selected KPI
                current_min = np.amin(feat_dataset_norm[:,:,i])

                # Normalize values corresponding to current selected KPI
                feat_dataset_norm[:,:,i] = (feat_dataset_norm[:,:,i] - current_min) / (_norm_thresh[j] - current_min)

            # Cap all values above 1
            feat_dataset_norm = np.minimum(feat_dataset_norm, 1)

            # Reshape into wide table
            feat_dataset_norm = feat_dataset_norm.reshape(feat_dataset_norm.shape[0], self.nb_hours * self.nb_kpis)

            return feat_dataset_norm

        self.kpiStats = kpiDesc()
        self.feat_norm = _feat_normalize(feat_dataset = self.feat_dataset)
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
                    "epochs" : 10,
                    "batch_size" : 64,
                    "encoding_dim": 5
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
        feat_dataset_norm_ = self.feat_norm.reshape(-1, self.nb_kpis, self.nb_hours, 1)

        # Data augmentation was used for the prediction model, hence enlarge images
        feat_dataset_norm_ = scipy.ndimage.zoom(feat_dataset_norm_, (1, 72/self.nb_kpis, 72/self.nb_hours, 1), order=0)

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
        df_kpi_cell = self.df_raw_norm_x_w 
        
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
