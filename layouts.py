# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:49:36 2022

@author: f_ati
"""

# Dash components, html, and dash tables
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
from click import style

import base64

# Import Bootstrap components
import dash_bootstrap_components as dbc
import numpy as np
# Import custom data.py
import data

# Import data from data.py file
df = data.df_final
df.drop(['order','positin1','round','wave','dec_o','position'], axis=1, inplace=True)


###################################################################### GRAPH #######################################################################################################

#Graph: Répartion de l'age : Valeurs atypiques sur l’age
fig = px.histogram(df, x="age",title='Répartion de l\'age')
fig.update_layout(bargap=0.2)
fig.update_layout({'plot_bgcolor':'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})

#Graph: Répartition des fréquences (go_out) de participation
fig2 = px.histogram(df, x="go_out",title='Répartition des fréquences (go_out) de participation')
fig2.update_layout({'plot_bgcolor':'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})


#Graph: Repartition de l'age en fonction du sexe selon le profile match
fig4 = px.box(df, x="gender", y="age",color="match",
    notched=True , # used notched shape
    title="Repartition de l'age en fonction du sexe selon le profile match")
fig4.update_layout({'plot_bgcolor':'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})



# Graph: Popularités des activités
def activite():
    df_activite =df.iloc[:,34:51]# df.iloc[:,40:57]
    df_activite['sports'] = (df_activite['sports'] + df_activite['exercise'] + df_activite['hiking'] + df_activite['yoga'])/4

    df_activite.drop(columns=['exercise','hiking','yoga'],inplace=True)

    df_activite['tv'] = (df_activite['tv'] + df_activite['tvsports'])/2

    df_activite.drop(columns=['tvsports'], inplace=True)

    df_activite = pd.DataFrame(df_activite.mean(axis=0)*10).reset_index()
    df_activite.columns = ['activity','mean']

    fig_activites = px.bar(df_activite, x='activity', y='mean', 
                    labels={'activity':'Activités','mean':'Popularité (en %)'}, 
                    color='mean', 
                    color_continuous_scale=['#FEF9E7','#F8C471','#F39C12','#9C640C'])


    fig_activites.update_layout({'plot_bgcolor':'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'},yaxis_range=[0,100])

    fig_activites.update_layout(title={'text':'Popularités des activités', 'y':0.9, 'x':0.5, 'xanchor':'center','yanchor':'top'})

    return fig_activites

#Graph: Salaire médian par tranche d\'age et par sexe
def income():
    
    income = df[['income','age', 'gender']]
    income.replace({',':''}, regex=True, inplace=True)
    income['tranche'] = pd.cut(income['age'], bins=np.arange(15,55,5), labels=['17-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])

    income = pd.DataFrame(income.groupby(['tranche', 'gender'])['income'].median()).reset_index()

    income.dropna(inplace=True)

   

    mediane_h = (income.groupby('gender')['income'].median())[1]
    mediane_f = (income.groupby('gender')['income'].median())[0]


    fig_income = px.bar(income, x='tranche', y='income', color='gender', 
                    barmode='group', labels={'income': 'Salaire', 'tranche':'Tranches d\'âge'}, 
                    color_discrete_map={'Homme':'#FEF9E7', 'Femme':'#F39C12'}).add_hline(mediane_h).add_hline(mediane_f)

    fig_income.update_layout({'plot_bgcolor':'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig_income.update_layout(title={'text':'Salaire médian par tranche d\'age et par sexe', 'y':0.9, 'x':0.5, 'xanchor':'center','yanchor':'top'}, 
                        legend_title_text='Genre')
    return fig_income

#Call function
fig3=activite()
fig5=income()


# Main applicaiton menu
appMenu = html.Div([
    dbc.Row(
        [
            dbc.Col(html.H4(style={'text-align': 'center'}, children='Select Wave:'),
                xs={'size':'auto', 'offset':0}, sm={'size':'auto', 'offset':0}, md={'size':'auto', 'offset':3},
                lg={'size':'auto', 'offset':0}, xl={'size':'auto', 'offset':0}),
            dbc.Col(dcc.Dropdown(
                style = {'text-align': 'center', 'font-size': '18px', 'width': '210px'},
                id='era-dropdown',
                
                clearable=False),
                xs={'size':'auto', 'offset':0}, sm={'size':'auto', 'offset':0}, md={'size':'auto', 'offset':0},
                lg={'size':'auto', 'offset':0}, xl={'size':'auto', 'offset':0}),

            dbc.Col(html.H4(style={'text-align': 'center', 'justify-self': 'right'}, children='Select Gender:'),
                xs={'size':'auto', 'offset':0}, sm={'size':'auto', 'offset':0}, md={'size':'auto', 'offset':3},
                lg={'size':'auto', 'offset':0}, xl={'size':'auto', 'offset':1}),
            dbc.Col(dcc.Dropdown(
                style = {'text-align': 'center', 'font-size': '18px', 'width': '210px'},
                id='team-dropdown',
                clearable=False),
                xs={'size':'auto', 'offset':0}, sm={'size':'auto', 'offset':0}, md={'size':'auto', 'offset':0},
                lg={'size':'auto', 'offset':0}, xl={'size':'auto', 'offset':0})
        ],
        form=True,
    ),
    
],className='menu')

# Layout for Analysis page
teamLayout = html.Div([
    
    ### Graphs of Historical  statistics ###
    dbc.Row(dbc.Col(html.H3(children='Analysis'))),
    
    dcc.Graph(
     
       figure=fig
   ),
     dcc.Graph(
       
        figure=fig2
    ),
    dcc.Graph(
      
        figure=fig3
    ),
    dcc.Graph(
     
        figure=fig4
    ),
    dcc.Graph(
     
        figure=fig5
    )
    
    # Bar Chart of Wins and Losses

],className='app-page')

image_homme = 'assets/profil.png'
image_femme = 'assets/femelle.png'
# Layout for profil page
profilLayout = html.Div(children=[
                html.Div(children=[
                    html.Img(src=image_homme, style={'position':'relative', 'top':'200px', 'height':'100px'}),
                    html.H1(children='27 ans', style={'position':'relative','left':'150px','top':'100px', 'font-size':'25px'}),
                    html.H1(children='Études de finance', style={'position':'relative','left':'150px','top':'105px','font-size':'25px'}),
                    html.H1(children='Caucasien', style={'position':'relative','left':'150px','top':'110px','font-size':'25px'}),
                    html.H1(children='Ingénieur', style={'position':'relative','left':'150px','top':'115px','font-size':'25px'}),
                    html.H1(children='46 274$/mois', style={'position':'relative','left':'150px','top':'120px','font-size':'25px'}),
                    html.H1(children='Aime le sport/les livres/les films/la musique', style={'position':'relative','left':'150px','top':'125px','font-size':'25px'}),
                    html.H1(children='1 date/mois', style={'position':'relative','left':'150px','top':'130px','font-size':'25px'}),
                    html.H1(children='2 sorties/semaine', style={'position':'relative','left':'150px','top':'135px','font-size':'25px'})
                ], style={'float':'left', 'width':'40%'}),

                html.Div(children=[
                    html.Img(src=image_femme, style={'position':'relative', 'top':'200px', 'height':'100px'}),
                    html.H1(children='26 ans', style={'position':'relative','left':'150px','top':'100px','font-size':'25px'}),
                    html.H1(children='Études de finance', style={'position':'relative','left':'150px','top':'105px','font-size':'25px'}),
                    html.H1(children='Sud-américain/Hispanique', style={'position':'relative','left':'150px','top':'110px','font-size':'25px'}),
                    html.H1(children='Ingénieur', style={'position':'relative','left':'150px','top':'115px','font-size':'25px'}),
                    html.H1(children='44 577$/mois', style={'position':'relative','left':'150px','top':'120px','font-size':'25px'}),
                    html.H1(children='Aime les diners/les musées/les films/les arts', style={'position':'relative','left':'150px','top':'125px','font-size':'25px'}),
                    html.H1(children='1 date/mois', style={'position':'relative','left':'150px','top':'130px','font-size':'25px'}),
                    html.H1(children='2 sorties/semaine', style={'position':'relative','left':'150px','top':'135px','font-size':'25px'})
                ], style={'float':'left','width':'40%'})
                ], style={'position':'absolute', 'top':'10px','left':'290px', 'width':'90%'})



    # Bar Chart of Wins and Losses

#],className='app-page')
