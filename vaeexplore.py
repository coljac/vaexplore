import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import argparse

def import_tensorflow():
    import tensorflow as tf

def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import torch.optim as optim
    from torchsummary import summary
    print(f"Torch version {torch.__version__}")

vae_dir = "/home/coljac/dev/pytorch/vae"
sys.path.append(vae_dir)
from newvae import LinearVAE, validate, fit


LATENT_DIMS = 3
FEATURES = 500

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("VAE", external_stylesheets=external_stylesheets)

config = {}

def init():
    if 'model' not in config:
        load_vae()
        load_test_set()

def load_vae(latent_dims=LATENT_DIMS, features=500):
    model = LinearVAE(latent_dims, features)
    model.load_state_dict(torch.load(vae_dir + '/vae.torch'))
    config['model'] = model.cuda()

def load_test_set():
    testdata = np.load(vae_dir + "/test_set.npz")
    config['x_test'] = testdata['x_test']
    config['y_test'] = testdata['y_test']
    config['outliers'] = testdata['outliers']
    config['x'] = testdata['x']

def y_axis_values():
    return [{'label': str(x+1), 'value': x+1} for x in range(1, LATENT_DIMS)],

init()
app.layout = html.Div([
    html.Div(["VAE Explorer"], 
        style={"font-size": "200%", "text-color": "red", "text-align": "center"}),
    html.Div([
        dcc.Graph(
            id='latent-scatter',
            # hoverData={'points': [{'customdata': 'Japan'}]}
        ),

        html.Div("X axis:",
            style={"float": "left", "padding-right": "10px"}),
        html.Div([
            dcc.Dropdown(
                id='latent-x-axis',
                options=[{'label': str(x+1), 'value': x+1} for x in range(LATENT_DIMS)],
                value='1'
                )], 
            style={"float": "left"}),

        html.Div("Y axis:", 
            style={"float": "left", "padding-right": "10px"}),
        html.Div([
            dcc.Dropdown(
                id='latent-y-axis',
                options=[{'label': str(x+1), 'value': x+1} for x in range(LATENT_DIMS)],
                # options=y_axis_values(),
                value='2'
            )], style={"float": "left"}),
        html.Div("Color code by variable:", 
            style={"float": "left", "padding-left": "10px", "padding-right": "10px"}),
        html.Div([
            dcc.RadioItems(
                id='color-code',
                options=[{'label': str(x+1), 'value': x} for x in range(config['y_test'].shape[1])],
                value=0)], style={"float": "left"}),
             # style={'width': '20%', 'display': 'inline-block', 'float': 'right',
                # 'border': 'thin blue solid'}),

    ], style={'width': '59%', 'display': 'inline-block', 'padding': '0 20',
        # 'border': 'thin red solid', 
        "float": "left"}),

    html.Div([
        dcc.Graph(id='single-example'),
        # dcc.Graph(id='reconstructed-example'),
    ], style={'display': 'inline-block', 'width': '39%',
        # 'border': 'thin green solid', 
        "float": "right"}),

    # html.Div(dcc.Slider(
        # id='crossfilter-year--slider',
        # min=df['Year'].min(),
        # max=df['Year'].max(),
        # value=df['Year'].max(),
        # marks={str(year): str(year) for year in df['Year'].unique()},
        # step=None
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


# Set color coding
# Set latent variables

@app.callback(
        Output('latent-scatter', 'figure'),
        [
            Input('latent-x-axis', 'value'),
            Input('latent-y-axis', 'value'),
            Input('color-code', 'value')
        ])
def update_graph(xaxis_dim, yaxis_dim, color_val):
    color_val = int(color_val)
    z = encode(torch.Tensor(config['x_test'][0:1000]))
    xaxis_dim = int(xaxis_dim)-1
    yaxis_dim = int(yaxis_dim)-1
    lx = z[:, xaxis_dim]
    ly = z[:, yaxis_dim]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lx,
                           y=ly,
                           marker={"color": config['y_test'][:, color_val],
                               "colorscale": 'viridis',
                               },
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    mode='markers',
                    name='Latent dim'))

    z = encode(torch.Tensor(config['outliers']))
    lx = z[:, xaxis_dim]
    ly = z[:, yaxis_dim]
    fig.add_trace(go.Scatter(x=lx,
                           y=ly,
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    # color='red',
                    mode='markers',
                    name='Latent dim'))
    fig.update_layout({"height": 700, "width": 900})
    fig.update_layout(showlegend=False)
    fig.update_layout(margin={"r": 0, 't': 0, 'b': 20})
    # legend=dict(
        # orientation="h",
        # yanchor="bottom",
        # y=1.02,
        # xanchor="right",
        # x=1
    # )) 
    return fig
    # markers = "s*o^x.")
    # for i in range(len(outliers)):
        # plt.scatter(ox[i], oy[i], s=50, color="red", marker=markers[i])# c=np.arange(len(outliers)), 
    # plt.colorbar()

@app.callback(
        Output('single-example', 'figure'),
        [Input('latent-scatter', 'hoverData')])
def update_single(hoverData):
    fig = go.Figure()

    if hoverData is None:
        return fig


    hoverData = hoverData['points'][0]

    if hoverData['curveNumber'] == 0:
        example = config['x_test'][hoverData['pointNumber']]
    else:
        example = config['outliers'][hoverData['pointNumber']]

    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))

    fig.add_trace(go.Scatter(x=config['x'],
                           y=example,
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    mode='lines',
                    name='Data'))
    z = transform(example)
    fig.add_trace(go.Scatter(x=config['x'],
                           y=z,
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    mode='lines',
                    name='Reconstruction'))
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )) 
    fig.update_layout({"height": 400, "width": 650})
    fig.update_layout(margin={"l": 0, "t": 0})
    return fig


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model-class', type=str, help='VAE wrapper class',
            # required=True)
    # args = parser.parse_args()
    # config['model'] = args.model_class
    # getattr(sys.modules[__name__], str)

    app.run_server(host="0.0.0.0", debug=True)

class VAEExplore(object):
    def __init__(self, latent_dims, features, mode="1D"):
        self.mode = mode
        self.latent_dims = latent_dims
        self.features = features

    def run(self, host="0.0.0.0", port=8050):
        app.run_server(host="0.0.0.0", debug=True)

    """
    Translate a test example into the latent space.
    """
    def encode(self, examples):
        print("Implement this.")

    """
    Given an example/examples, encode then decode to show the VAE's reconstruction.
    """
    def transform(example):
        model = config['model']
        result = model(torch.Tensor(example[np.newaxis, :]).cuda())
        result = result[0].cpu().data.numpy()[0]
        return result

    """
    Fetch the test set for plotting.
    """
    def test_set():
        return np.random.random((100, self.features))

    """
    Fetch a second test set for plotting, to be flagged as outliers.
    """
    def outliers():
        return None

    def y_axis_values(self):
        return [{'label': str(x+1), 'value': x+1} for x in range(1, self.latent_dims)],

