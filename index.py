# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:47:41 2022

@author: f_ati
"""

# import dash-core, dash-html, dash io, bootstrap
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from PIL import Image
# Dash Bootstrap components
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
# Navbar, layouts, custom callbacks
from layouts import profilLayout ,teamLayout

# Import custom data.py
import data

# Import app
from app import app
# Import server for deployment
from app import srv as server
# Import data from data.py file
df=data.df_final


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
logo1=Image.open("assets/logo2.png")
sidebar = html.Div(
    [
        #html.H2("AI MATCH", className="display-5"),
        html.Img(src=logo1, height="200",width="230"),
        html.H2("Data Explorer", className="display-5"),
        html.Hr(),
        dbc.Nav(
            [dbc.NavLink("Home", href="/", active="exact")],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H2("Historical Analysis", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Analysis", href="/Analysis", active="exact"),
                dbc.NavLink("Profiles of participants", href="/Profiles", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H2("Machine Learning Analysis", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Projections and Regression", href="/projection", active="exact"),
                # dbc.NavLink("Batting Analysis", href="/player", active="exact"),
                # dbc.NavLink("Pitching/Feilding Analysis", href="/field", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

# Sidebar layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

logo=Image.open("assets/logo.png")

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == '/':
        return [html.Div([dcc.Markdown('''
            ###  TROUVER UNE PERSONNE PARFAITEMENT COMPATIBLE
            
            Cette application est un projet de Easy Date réalisé par [Haidara Fatimetou, KIEMDE Christelle, DUBRULLE Pierre] 
            à l'aide de Dash de Plotly,
            les composants Dash Bootstrap de faculty.ai, Pandas,et des fonctions personnalisées. 
    
            Easy Date est une société d'événementiel qui organise des speed dating.
            Pour s’inscrire, une personne doit remplir un formulaire avec différentes
            informations sur elle et ses attentes. L’entreprise récolte ainsi les données et
            organise différentes vagues de speed dating.
            
            Lors d’une session, un participant rencontre plusieurs personnes. A l’issue tr(jh,,,,)
            chaque rencontre, il décide si Oui/Non la personne veut revoir un ou des
            coups de cœur.
            
            Le problème, c’est le faible taux de match qui fait perdre beaucoup de temps
            à l’entreprise et donc de l’argent.
            
            L’équipe de data scientist de l’entreprise doit donc réfléchir à un modèle
            permettant de prédire si deux personnes vont matcher selon le formulaire
            complété préalablement de la rencontre.
            
            Cela fait des mois qu’ils planchent sur le projet, ils ont déjà calculé un score de
            similarité entre deux participants en se basant sur leurs réponses aux
            questionnaires. 
            Cependant, ce score ne semble pas fiable et l’entreprise
            souhaiterait un modèle de scoring plus performant pour prédire si l’amour va
            opérer entre deux personnes.
            
            C’est la raison pour laquelle le projet AI match a vu le jour !
        ''')],className='home',style={'text-align':'center'}),
        
        html.Div(html.Img(src=logo), style={'text-align':'center'}),
        
        
        
        ]
    elif pathname == '/Analysis':
        return teamLayout
    elif pathname == '/Profiles':
        return profilLayout
    
    else:
        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )




# Call app server
if __name__ == '__main__':
    # set debug to false when deploying app
    app.run_server(debug=True)
