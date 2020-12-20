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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("VAE", external_stylesheets=external_stylesheets)

config = {}

class Model(object):
    def __init__(self, encoder, transformer, latent_dims, x, test_set, y_test, outliers=None):
        self.outliers = outliers
        self.test_set = test_set
        self.x = x
        self.latent_dims = latent_dims
        self.y_test = y_test
        self.encoder = encoder
        self.transformer = transformer
        config['model'] = self

    def encode(self, examples):
        return self.encoder(examples)

    def reconstruct(self, examples):
        return self.transformer(examples)

    def run(self):
        init_ui()
        app.run_server(host="0.0.0.0", debug=True)

def init_ui():
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
                    options=[{'label': str(x+1), 'value': x+1} for x in range(config['model'].latent_dims)],
                    value='1'
                    )], 
                style={"float": "left"}),

            html.Div("Y axis:", 
                style={"float": "left", "padding-right": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id='latent-y-axis',
                    options=[{'label': str(x+1), 'value': x+1} for x in range(config['model'].latent_dims)],
                    # options=y_axis_values(),
                    value='2'
                )], style={"float": "left"}),
            html.Div("Color code by variable:", 
                style={"float": "left", "padding-left": "10px", "padding-right": "10px"}),
            html.Div([
                dcc.RadioItems(
                    id='color-code',
                    options=[{'label': str(x+1), 'value': x} for x in range(config['model'].y_test.shape[1])],
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

    ])

@app.callback(
        Output('latent-scatter', 'figure'),
        [
            Input('latent-x-axis', 'value'),
            Input('latent-y-axis', 'value'),
            Input('color-code', 'value')
        ])
def update_graph(xaxis_dim, yaxis_dim, color_val):
    model = config['model']
    color_val = int(color_val)
    # z = model.encode(torch.Tensor(model.test_set[0:1000]))
    z = model.encode(model.test_set[0:2000])
    xaxis_dim = int(xaxis_dim)-1
    yaxis_dim = int(yaxis_dim)-1
    lx = z[:, xaxis_dim]
    ly = z[:, yaxis_dim]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lx,
                           y=ly,
                           marker={"color": model.y_test[:, color_val],
                               "colorscale": 'viridis',
                               },
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    mode='markers',
                    name='Latent dim'))

    if model.outliers is not None:
        z = model.encode(model.outliers)
        lx = z[:, xaxis_dim]
        ly = z[:, yaxis_dim]

        symbols = "circle,square,diamond,cross,triangle-up,star,hexagram,star-square,triangle-down".split(",")

        fig.add_trace(go.Scatter(x=lx,
                               y=ly,
                        mode='markers',
                        marker_symbol=symbols,
                        marker_line_color="#AA0000",
                        marker_line_width=2,
                        marker=dict(color='#EE3333', size=10),
                        name='Latent dim'))


    fig.update_layout({"height": 700, "width": 900})
    fig.update_layout(showlegend=False)
    fig.update_layout(margin={"r": 0, 't': 0, 'b': 20})
    return fig

@app.callback(
        Output('single-example', 'figure'),
        [Input('latent-scatter', 'hoverData')])
def update_single(hoverData):
    fig = go.Figure()
    model = config['model']

    if hoverData is None:
        return fig


    hoverData = hoverData['points'][0]

    if hoverData['curveNumber'] == 0:
        example = model.test_set[hoverData['pointNumber']]
    else:
        example = model.outliers[hoverData['pointNumber']]

    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))

    fig.add_trace(go.Scatter(x=model.x,
                           y=example,
                    # hovertemplate = '%{x:.1f} years<br>\nPrinciple: %{y:$,.0f}<extra></extra>',
                    mode='lines',
                    name='Data'))
    z = model.reconstruct(example)
    fig.add_trace(go.Scatter(x=model.x,
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

