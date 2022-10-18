## Importation des librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.impute import KNNImputer
from scipy.stats import mode
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,make_scorer,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.preprocessing import Imputer

## Importation de la base de données
df=pd.read_csv("C:\\Users\\USER\\Documents\\Master_SISE\\Projet\\Projet_python\\aimatch\\train.csv",sep=';')
submiss=pd.read_csv("C:\\Users\\USER\\Documents\\Master_SISE\\Projet\\Projet_python\\aimatch\\submissions.csv",sep=";")
print(df.shape)
print(submiss.shape)
 
 ## Tester si les deux bases ont les memes colonnes
df.equals(submiss)
## Recupération des variables différentes
df.columns.difference(submiss.columns)
## Suppressions des variables différentes
df=df.drop(['order','positin1','round','wave','dec_o','position'], axis=1)

## description des données
df.info()
df.career_c.value_counts()
## Nettoyage de données
## Changement du type de variable
df["shar1_1"]=df["shar1_1"].str.replace(',', '.')
df["attr1_1"]=df["attr1_1"].str.replace(',', '.')
df["sinc1_1"]=df["sinc1_1"].str.replace(',', '.')
df["intel1_1"]=df["intel1_1"].str.replace(',', '.')
df["fun1_1"]=df["fun1_1"].str.replace(',', '.')
df["amb1_1"]=df["amb1_1"].str.replace(',', '.')
df["income"]=df["income"].str.replace(',', '.')
df["pf_o_att"]=df["pf_o_att"].str.replace(',', '.')
df["pf_o_sin"]=df["pf_o_sin"].str.replace(',', '.')
df["pf_o_int"]=df["pf_o_int"].str.replace(',', '.')
df["pf_o_fun"]=df["pf_o_fun"].str.replace(',', '.')
df["pf_o_amb"]=df["pf_o_amb"].str.replace(',', '.')
df["pf_o_sha"]=df["pf_o_sha"].str.replace(',', '.')
df["int_corr"]=df["int_corr"].str.replace(',', '.')
df.int_corr = df.int_corr.astype('float')
df.shar1_1 =df.shar1_1.astype('float')
df.attr1_1=df.shar1_1.astype('float')
df.sinc1_1=df.shar1_1.astype('float')
df.intel1_1=df.shar1_1.astype('float')
df.fun1_1=df.shar1_1.astype('float')
df.amb1_1=df.shar1_1.astype('float')
df.income=df.shar1_1.astype('float')
df.pf_o_att=df.pf_o_att.astype('float')
df.pf_o_sin=df.pf_o_sin.astype('float')
df.pf_o_int=df.pf_o_int.astype('float')
df.pf_o_fun=df.pf_o_fun.astype('float')
df.pf_o_amb=df.pf_o_amb.astype('float')
df.pf_o_sha=df.pf_o_sha.astype('float')
#df.go_out=df.go_out.astype('object')
df.field_cd=df.field_cd.astype('float')                                       
#df.career_c=df.career_c.astype('object')

## Scinder la base en quanti et quali pour le tyraitement des données manquantes
dfquanti=df.select_dtypes(exclude=['object'])
dfquali=df.select_dtypes(include=['object'])
## correction des données manquantes par les K plus proches voisins
imputer = KNNImputer(n_neighbors=3)
df2= pd.DataFrame(imputer.fit_transform(dfquanti),columns=dfquanti.columns)
df2.isna().any()
## Correction des données manquantes par le mode
for i in dfquali.columns :
  dfquali[i].fillna(dfquali[i].mode()[0], inplace=True)
print(dfquali)
dfquali.isna().any()
## fusion des deux bases 
df_final= pd.concat([df2,dfquali],axis=1)
df_final
## verification si données manquantes
print(df_final.isnull().sum())
## 
df_final.int_corr = df_final.int_corr.astype('int')
df_final.shar1_1 =df_final.shar1_1.astype('int')
df_final.attr1_1=df_final.shar1_1.astype('int')
df_final.sinc1_1=df_final.shar1_1.astype('int')
df_final.intel1_1=df_final.shar1_1.astype('int')
df_final.fun1_1=df_final.shar1_1.astype('int')
df_final.amb1_1=df_final.shar1_1.astype('int')
df_final.income=df_final.shar1_1.astype('int')
df_final.pf_o_att=df_final.pf_o_att.astype('int')
df_final.pf_o_sin=df_final.pf_o_sin.astype('int')
df_final.pf_o_int=df_final.pf_o_int.astype('int')
df_final.pf_o_fun=df_final.pf_o_fun.astype('int')
df_final.pf_o_amb=df_final.pf_o_amb.astype('int')
df_final.pf_o_sha=df_final.pf_o_sha.astype('int')
df_final.go_out=df_final.go_out.astype('int')
df_final.career_c=df_final.career_c.astype('int')
df_final.field_cd=df_final.field_cd.astype('int')

#labélisation des variables
# code0=[0,1] 
# mat=["no","yes"] 
# sex=["Female","Male"] 
# df_final['match'] = df_final['match'].replace(code0, mat) 
# df_final['samerace'] = df_final['samerace'].replace(code0, mat) 
# df_final['gender'] = df_final['gender'].replace(code0, sex) 
# code001=[1,2,3,4,5,6,7] 
# go=["Several times a week","Twice a wee",
# "Once a week","Twice a month","Once a month","Several times a year","Almost never"] 
# df_final['go_out'] = df_final['go_out'].replace(code001, go)

# code = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] 
# status = ["Law ", "Math", "Social Science,Psychologist" ,
#  "Medical Science, Pharmaceuticals, and Bio Tech ", "Engineering ", "English/Creative Writing/ Journalism ","History/Religion/Philosophy"," Business/Econ/Finance", "Education, Academia ","Biological Sciences/Chemistry/Physics","Social Work ","Undergrad/undecided", "Political Science/International Affairs","Film","Fine Arts/Arts Administration","Languages", "Architecture","Other"]

# df_final['field_cd'] = df_final['field_cd'].replace(code, status)

# code2=[1,2,3,4,5,6] 
# races=["Black/African American","European/Caucasian-American","Latino/Hispanic American", "Asian/Pacific Islander/Asian-American","Native American","Other"] 
# df_final['race']=df_final['race'].replace(code2,races)

# code3=[1,2,3,4,5,6] 
# goal1=["Seemed like a fun night out","To meet new people","To get a date","Looking for a serious relationship","To say I did it","Other"] 
# df_final['goal']=df_final['goal'].replace(code3,goal1)

# code4=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] 
# career_c1=["Lawyer ","Academic/Research ","Psychologist ","Doctor/Medicine ","Engineer","Creative Arts/Entertainment",
# "Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin", "Real Estate ","International/Humanitarian Affairs ",
# "Undecided ","Social Work","Speech Pathology","Politics","Pro sports/Athletics","Other","Journalism","Architecture"]

# df_final['career_c']=df_final['career_c'].replace(code4,career_c1)
## Traitement des doublons
df_final.describe(exclude="object")
#df_final.describe(exclude="object",percentiles=np.linspace(start = 0, stop = 1, num= 11))
df_final.describe(include="object")

##Répartition 0/1 sur la target (match)
df_final.match.value_counts()
df.match.value_counts(normalize=True)
## Répartition homme/femme
df_final.gender.value_counts()
##Répartition des fréquences (go_out) de participation
df_final.go_out.value_counts()

## Visualisation
## Modelisation
## Isoler la matrice des explicatives 
X=df_final[["gender","age","age_o","samerace","field_cd","income","go_out","career_c","pf_o_att","attr_o","goal"
,"imprace"]]
Y=df_final.match
X.shape
Y.shape
## Codage dijonctif complete
X=pd.get_dummies(X,columns=["go_out","career_c","income","field_cd"],drop_first=True)

##Scindez les données en échantillons d’apprentissage
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,train_size=0.70,stratify=Y,random_state=42)
## Vérifiez les dimensions des structures générées
print(Xtrain.shape)
print(Xtest.shape)
## Instanciez une nouvelle version de l’arbre de décision
arbre = DecisionTreeClassifier(random_state=0)
arbre.fit(Xtrain,Ytrain)
## Importances des variables
imp = {"VarName":Xtrain.columns,"Importance":arbre.feature_importances_}
tmpImp = pd.DataFrame(imp).sort_values(by="Importance",ascending=False)
print(tmpImp)


## Sélection des variables importantes à mettre dans le modèle
variables = tmpImp.loc[tmpImp['Importance'] > 0.03, 'VarName'].to_list()
variables
Xtrain=Xtrain[variables]
Xtest=Xtest[variables]


## On reprends le modèle avec les variables importantes
arbre1 = DecisionTreeClassifier(random_state=0)
arbre1.fit(Xtrain,Ytrain)
## Affichez l’arbre graphiquement
plot_tree(arbre1,feature_names=Xtrain.columns,filled=True)
## Prédiction
y_pred=arbre1.predict(Xtest)
y_pred
from sklearn.metrics import confusion_matrix
mc = pd.DataFrame(confusion_matrix(Ytest,y_pred), 
                  columns=['pred_0','pred_1'],
                  index=['obs_0','obs_1'])

mc
## Evaluation du modèle
print(accuracy_score(Ytest,y_pred))
## F1 score
print('f1_score : ' + 
      str(f1_score(Ytest,y_pred, average='macro')))


## GridSearchCV
## les paramètres 

parameters = {'max_depth' : np.arange(start = 1, stop = 10, step = 1) , 
              'min_samples_leaf' : np.arange(start = 5, stop = 250, step = 50),
              'min_samples_split' : np.arange(start = 10, stop = 500, step = 50)}

modele_arbre = DecisionTreeClassifier()
f1 = make_scorer(f1_score , average='macro')
modele_arbre = GridSearchCV(modele_arbre,
                                  parameters,
                                  scoring = f1,
                                  verbose = 2, 
                                  cv = 5)
modele_arbre.fit(Xtrain, Ytrain)

print("Voici les paramètres du meilleure modèle : " + 
      str(modele_arbre.best_estimator_))
print("Voici le "  + str(modele_arbre.scorer_) + 
      " du meilleure modèle : " + str(modele_arbre.best_score_))
## Prédiction sur les données de test
y_pred = modele_arbre.predict(Xtest)
## Evaluation
print('f1_score : ' + 
      str(f1_score(Ytest,y_pred, average='macro')))

## Modélisation 2
#visualisation des données

pd.value_counts(df_final['match']).plot.bar()
plt.title('match relation')
plt.xlabel('match')
plt.ylabel('Frequency')
df_final['match'].value_counts()

print("Before OverSampling, counts of label '1': {}".format(sum(Ytrain==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Ytrain==0)))
sm = SMOTE(random_state=2)
Xtrain_res, Ytrain_res = sm.fit_resample(Xtrain, Ytrain.ravel())
print('After OverSampling, the shape of train_X: {}'.format(Xtrain_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(Ytrain_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(Ytrain_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Ytrain_res==0)))
## Modèle avec les données équilibrées
parameters = {'max_depth' : np.arange(start = 1, stop = 10, step = 1) , 
              'min_samples_leaf' : np.arange(start = 5, stop = 250, step = 50),
              'min_samples_split' : np.arange(start = 10, stop = 500, step = 50)}

modele_arbre2 = DecisionTreeClassifier()
f1 = make_scorer(f1_score , average='macro')
modele_arbre2 = GridSearchCV(modele_arbre2,
                                  parameters,
                                  scoring = f1,
                                  verbose = 2, 
                                  cv = 5)
modele_arbre2.fit(Xtrain_res, Ytrain_res)

print("Voici les paramètres du meilleure modèle : " + 
      str(modele_arbre2.best_estimator_))
print("Voici le "  + str(modele_arbre2.scorer_) + 
      " du meilleure modèle : " + str(modele_arbre2.best_score_))

## Prédiction

y_pred = modele_arbre2.predict(Xtest)
## Evaluation
print('f1_score : ' + 
      str(f1_score(Ytest,y_pred, average='macro')))

## SBorderline SMOTE sur le même ensemble de données déséquilibré.
oversample = SVMSMOTE()
X, y = oversample.fit_resample(Xtrain, Ytrain)
parameters = {'max_depth' : np.arange(start = 1, stop = 10, step = 1) , 
              'min_samples_leaf' : np.arange(start = 5, stop = 250, step = 50),
              'min_samples_split' : np.arange(start = 10, stop = 500, step = 50)}

modele_arbre3 = DecisionTreeClassifier()
f1 = make_scorer(f1_score , average='macro')
modele_arbre3 = GridSearchCV(modele_arbre3,
                                  parameters,
                                  scoring = f1,
                                  verbose = 2, 
                                  cv = 5)
modele_arbre3.fit(X, y)
print("Voici les paramètres du meilleure modèle : " + 
      str(modele_arbre3.best_estimator_))
print("Voici le "  + str(modele_arbre3.scorer_) + 
      " du meilleure modèle : " + str(modele_arbre3.best_score_))


## Prédiction
y_pred = modele_arbre3.predict(Xtest)
## Evaluation
print('f1_score : ' + 
      str(f1_score(Ytest,y_pred, average='macro')))






## Modélisation avec les K plus proches voisins
neigboars=np.arange(1,20)

train_accuracy=[]
test_accuracy=[]

for i,k in enumerate(neigboars):
    knn=KNeighborsClassifier(n_neighbors=k)
    #fitting
    knn.fit(Xtrain_res,Ytrain_res)
    
    train_accuracy.append(knn.score(Xtrain_res,Ytrain_res))
    test_accuracy.append(knn.score(Xtest,Ytest))

   #visualization

plt.figure(figsize=[15,10])
plt.plot(neigboars,test_accuracy,label="Test accuracy")
plt.plot(neigboars,train_accuracy,label="Train accuracy")
plt.legend()

plt.title("ACCURACY RATE")
plt.xlabel("Number of Neighboars")
plt.ylabel("Accuracy")
plt.xticks(neigboars)
plt.show()

#Trouvons la meilleure précision pour laquelle la valeur de k
print("Best Accuracy is {} with k= {}".format(np.max(test_accuracy),
1+test_accuracy.index(np.max(test_accuracy))))
## On refait le modèle avec K =2
knn = KNeighborsClassifier(n_neighbors=2)
#Train the model using the training sets
knn.fit(Xtrain_res, Ytrain_res)

#Predict the response for test dataset
y_pred4 = knn.predict(Xtest)
## Evaluation
print('f1_score : ' + 
      str(f1_score(Ytest,y_pred4, average='macro')))

## Compétition kaggle
## Traitement sur la base submission

## Scinder la base en quanti et quali pour le tyraitement des données manquantes
dfqusuantii=submiss.select_dtypes(exclude=['object'])
dfqualif=submiss.select_dtypes(include=['object'])
dfqusuantii=dfqusuantii.drop(['expnum'], axis=1)
print(dfqusuantii)
## correction des données manquantes par les K plus proches voisins
imputer = KNNImputer(n_neighbors=3)
dff= pd.DataFrame(imputer.fit_transform(dfqusuantii),columns=dfqusuantii.columns)
dff.isna().any()
## Correction des données manquantes par le mode
for i in dfqualif.columns :
  dfqualif[i].fillna(dfqualif[i].mode()[0], inplace=True)
print(dfqualif)
dfqualif.isna().any()
## fusion des deux bases 
sumiss_final= pd.concat([dff,dfqualif],axis=1)
## verification si données manquantes
print(sumiss_final.isnull().sum())
sumiss_final=sumiss_final[variables]
sumiss_final.replace(",","",regex=True,inplace=True)
print(sumiss_final)
## Predire sur les données submiss
## Prédiction
#y_pred2 = modele_arbre2.predict(Xtest)
y_pred_submiss = modele_arbre2.predict(sumiss_final)
y_pred_sumiss=pd.DataFrame(y_pred_submiss)
y_pred_sumiss.describe()
y_pred_sumiss.to_csv("scores.csv",",")


## Interpretation du modèle: l'influence des variables sur le meilleur modèle
imp = {"VarName":Xtrain.columns,"Importance":modele_arbre2.best_estimator_.feature_importances_}
tmpImp = pd.DataFrame(imp).sort_values(by="Importance",ascending=False)
print(tmpImp)
