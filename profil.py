# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:26:37 2022

@author: USER
"""

# -*- coding: utf-8 -*-
from click import style
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import base64

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

image_homme = 'assets/profil.png'
image_femme = 'assets/femelle.png'

app.layout = html.Div(children=[
                html.Div(children=[
                    html.Img(src=image_homme, style={'position':'relative', 'top':'200px', 'height':'100px'}),
                    html.H1(children='27 ans', style={'position':'relative','left':'150px','top':'100px'}),
                    html.H1(children='Études de finance', style={'position':'relative','left':'150px','top':'105px'}),
                    html.H1(children='Caucasien', style={'position':'relative','left':'150px','top':'110px'}),
                    html.H1(children='Ingénieur', style={'position':'relative','left':'150px','top':'115px'}),
                    html.H1(children='46 274$/mois', style={'position':'relative','left':'150px','top':'120px'}),
                    html.H1(children='Aime le sport, les livres, les films et la musique', style={'position':'relative','left':'150px','top':'125px'}),
                    html.H1(children='1 date/mois', style={'position':'relative','left':'150px','top':'130px'}),
                    html.H1(children='2 sorties/semaine', style={'position':'relative','left':'150px','top':'135px'})
                ], style={'float':'left', 'width':'50%'}),

                html.Div(children=[
                    html.Img(src=image_femme, style={'position':'relative', 'top':'200px', 'height':'100px'}),
                    html.H1(children='26 ans', style={'position':'relative','left':'150px','top':'100px'}),
                    html.H1(children='Études de finance', style={'position':'relative','left':'150px','top':'105px'}),
                    html.H1(children='Sud-américain/Hispanique', style={'position':'relative','left':'150px','top':'110px'}),
                    html.H1(children='Ingénieur', style={'position':'relative','left':'150px','top':'115px'}),
                    html.H1(children='44 577$/mois', style={'position':'relative','left':'150px','top':'120px'}),
                    html.H1(children='Aime les diners, les musées, les films et les arts', style={'position':'relative','left':'150px','top':'125px'}),
                    html.H1(children='1 date/mois', style={'position':'relative','left':'150px','top':'130px'}),
                    html.H1(children='2 sorties/semaine', style={'position':'relative','left':'150px','top':'135px'})
                ], style={'float':'right','width':'50%'})
                ], style={'position':'fixed', 'width':'100%', 'margin':'auto 50px auto 50px'})




if __name__ == '__main__':
    app.run_server(debug=True)