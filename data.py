# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:39:23 2022

@author: f_ati
"""


# Import Pandas
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import dash
import dash_core_components as dcc
import dash_html_components as html
data = pd.read_csv("train.csv",sep=";")




data["shar1_1"]=data["shar1_1"].str.replace(',', '.')
data["attr1_1"]=data["attr1_1"].str.replace(',', '.')
data["sinc1_1"]=data["sinc1_1"].str.replace(',', '.')
data["intel1_1"]=data["intel1_1"].str.replace(',', '.')
data["fun1_1"]=data["fun1_1"].str.replace(',', '.')
data["amb1_1"]=data["amb1_1"].str.replace(',', '.')
data["income"]=data["income"].str.replace(',', '.')
data["pf_o_att"]=data["pf_o_att"].str.replace(',', '.')
data["pf_o_sin"]=data["pf_o_sin"].str.replace(',', '.')
data["pf_o_int"]=data["pf_o_int"].str.replace(',', '.')
data["pf_o_fun"]=data["pf_o_fun"].str.replace(',', '.')
data["pf_o_amb"]=data["pf_o_amb"].str.replace(',', '.')
data["pf_o_sha"]=data["pf_o_sha"].str.replace(',', '.')
data["int_corr"]=data["int_corr"].str.replace(',', '.')

data.int_corr = data.int_corr.astype('float')
data.shar1_1 =data.shar1_1.astype('float')
data.attr1_1=data.shar1_1.astype('float')
data.sinc1_1=data.shar1_1.astype('float')
data.intel1_1=data.shar1_1.astype('float')
data.fun1_1=data.shar1_1.astype('float')
data.amb1_1=data.shar1_1.astype('float')
data.income=data.shar1_1.astype('float')
data.pf_o_att=data.pf_o_att.astype('float')
data.pf_o_sin=data.pf_o_sin.astype('float')
data.pf_o_int=data.pf_o_int.astype('float')
data.pf_o_fun=data.pf_o_fun.astype('float')
data.pf_o_amb=data.pf_o_amb.astype('float')
data.pf_o_sha=data.pf_o_sha.astype('float')
data.go_out=data.go_out.astype('object')
data.dtypes

# Treatment of missing values
dfquati=data.select_dtypes(exclude=['object'])
dfquati

imputer = KNNImputer(n_neighbors=2)
df2= pd.DataFrame(imputer.fit_transform(dfquati),columns=dfquati.columns)
df2=df2.astype("int")
df2.isna().any()

print(data.info())
print(data.shape)

dfquali=data.select_dtypes(include=['object'])
dfquali

df_final= pd.concat([df2,dfquali],axis=1)
#Correction of missing data by the mode
for i in dfquali.columns :
  dfquali[i].fillna(dfquali[i].mode()[0], inplace=True)
print(dfquali)
dfquali.isna().any()

# Merger of the two bases
df_final= pd.concat([df2,dfquali],axis=1)
df_final

print(df_final.isnull().sum())

# Labelling of variables
code0=[0,1]
mat=["no","yes"]
sex=["Female","Male"]
df_final['match'] = df_final['match'].replace(code0, mat)
df_final['samerace'] = df_final['samerace'].replace(code0, mat)
df_final['gender'] = df_final['gender'].replace(code0, sex)
code001=[1,2,3,4,5,6,7]
go=["Several times a week","Twice a wee","Once a week","Twice a month","Once a month","Several times a year","Almost never"]
df_final['go_out'] = df_final['go_out'].replace(code001, go)

code = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
status = ["Law ", "Math", "Social Science,Psychologist" , "Medical Science, Pharmaceuticals, and Bio Tech ",
          "Engineering ", "English/Creative Writing/ Journalism ","History/Religion/Philosophy"," Business/Econ/Finance",
          "Education, Academia ","Biological Sciences/Chemistry/Physics","Social Work ","Undergrad/undecided",
          "Political Science/International Affairs","Film","Fine Arts/Arts Administration","Languages",
          "Architecture","Other"]

df_final['field_cd'] = df_final['field_cd'].replace(code, status)


code2=[1,2,3,4,5,6]
races=["Black/African American","European/Caucasian-American","Latino/Hispanic American",
       "Asian/Pacific Islander/Asian-American","Native American","Other"]
df_final['race']=df_final['race'].replace(code2,races)


code3=[1,2,3,4,5,6]
goal1=["Seemed like a fun night out","To meet new people","To get a date","Looking for a serious relationship","To say I did it","Other"]
df_final['goal']=df_final['goal'].replace(code3,goal1)

code4=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
career_c1=["Lawyer ","Academic/Research ","Psychologist ","Doctor/Medicine ","Engineer","Creative Arts/Entertainment","Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin",
           "Real Estate ","International/Humanitarian Affairs ","Undecided ","Social Work","Speech Pathology","Politics","Pro sports/Athletics","Other","Journalism","Architecture"]

df_final['career_c']=df_final['career_c'].replace(code4,career_c1)

