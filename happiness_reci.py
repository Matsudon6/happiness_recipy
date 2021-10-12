# Importation des packages nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable





########################
# Links original datas #
########################

links = {}

for i in range (6):
	links["{}".format(2015+i)] = "https://www.kaggle.com/mathurinache/world-happiness-report?select={}.csv".format(2015+i)

links["2021"] = "https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021?select=world-happiness-report-2021.csv"




#####################
# Turn off warnings #
#####################

st.set_option('deprecation.showPyplotGlobalUse', False)




########################
# Chargement des datas #
########################

@st.cache
def load_csv(path):
	return pd.read_csv('datas/'+path, index_col=0)

df = load_csv('whr_2015_2021_scaled_dummies.csv')
dfr = load_csv('whr_2015_2021_scaled.csv')
scores = load_csv('scores.csv')


# Préparation des ensembles d'entraînement et de test
# Ensemble d'entraînement
X_train = df.loc[df['year'] < 2020].drop(['country', 'score', 'year'], axis=1)
y_train = df.loc[df['year'] < 2020, 'score']

# Ensemble de test
X_test = df.loc[df['year'] >= 2020].drop(['country', 'score', 'year'], axis=1)
y_test = df.loc[df['year'] >= 2020, 'score']





##########################
# Chargement des modèles #
##########################

# Régression linéaire
lr = LinearRegression()
lr.fit(X_train,y_train)


# RFE
rfe = RFE(estimator=lr, n_features_to_select=4, step=1)
rfe.fit(X_train, y_train)


# RidgeCV
alphas = [0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100]
ridge2 = RidgeCV(alphas, cv=3)
ridge2.fit(X_train.iloc[:,:4], y_train)


# LassoCV
alphas = (0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0)
lasso_cv = LassoCV(alphas=alphas)
lasso_cv.fit(X_train, y_train)


# Random Forest #1
rf1 = RandomForestRegressor(oob_score=True, max_depth=None, n_estimators=100, n_jobs=-1)
rf1.fit(X_train, y_train)




# Random Forest #2
rf2 = RandomForestRegressor(oob_score=True, 
                           n_estimators=100, # Valeur par défaut
                           max_depth=3, # Pour éviter l'overfitting
                           n_jobs=-1) # Pour utiliser au max le CPU

rf2.fit(X_train.iloc[:,1:4], y_train)



# Random Forest #3
rf3 = RandomForestRegressor(
		max_depth=6, 
		oob_score=True,
		n_estimators=100, 
		n_jobs=-1) 

rf3.fit(X_train.iloc[:,:4], y_train)






#############
# Fonctions #
#############

# Fonction pour afficher les scatterplot des scores vs…
def vs_score_plot(x, title, col):
	fig = plt.figure(figsize=(4,4))#, facecolor='white'
	ax = fig.add_subplot(111)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["top"].set_visible(False)
	ax.set_xlim([-.1,1.1])
	ax.set_title(title, fontsize=14)
	ax.grid(alpha=.5)
	sns.regplot(x=x, y=df['score'], line_kws={'color': '#EA0B46'}, scatter_kws={'alpha': .6, 'color':'#186E8B'});
	
	col.pyplot(fig)


# Fonction pour l'affichage des scores
#@st.cache
def plot_scores(oob = False, train=[0,0,0], test=[0,0,0]):
    # Inititalisation du compteur
    count = 0

    plot_titre = ['R2', 'MSE', 'MAPE']

    # Instanciation de la figure
    fig = plt.figure(figsize=(10,3))
    #fig.set_facecolor('white')

    for i in np.arange (1,4):
        ax = fig.add_subplot(1,3,count+1)
        ax.set_facecolor('#F0F2F6')

        maxi = max(train[count], test[count]) * 1.2
        mini = min(train[count], test[count])
        mini = 0 if mini >= 0 else mini 

        sns.barplot(x=[0,1],y=[train[count], test[count]], palette=['#186E8B', '#EA0B46'])#, color

        if (count == 0) & (oob == True) : 
        	ax.set_xticklabels(['OOB', 'Test'], fontsize=12)
        else:
        	ax.set_xticklabels(['Train', 'Test'], fontsize=12)
        
        if count == 1:
        	ax.set_ylim([0,1])
        else:
        	ax.set_ylim([mini,maxi])

        if count == 2:
        	ax.set_ylim([0,11.5])
          
        ax.set_xlabel(plot_titre[count], fontsize=14, fontweight="bold")
        
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.text(0,train[count]*1.05, train[count], color='black', ha="center", fontsize=16)
        ax.text(1,test[count]*1.05, test[count], color='black', ha="center", fontsize=16, fontweight="bold")

        count +=1;


# Fonction pour le calcul de la MAPE
def MAPE(y_true, y_pred):
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Fonction pour l'affichage des différentes métriques
def print_perf(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

	# On vérifie s'il y a un score OOB
	if hasattr(model, 'oob_score_'): 
		r2_train = round(model.oob_score_, 2)
		oob = True
	else:
		r2_train = round(model.score(X_train, y_train),2)
		oob = False

	# On calcule les autres scores
	r2_test = round(model.score(X_test, y_test),2)
	mse_train = round(mean_squared_error(y_train, model.predict(X_train)),2)
	mse_test = round(mean_squared_error(y_test, model.predict(X_test)),2)
	mape_train = round(MAPE(y_train, model.predict(X_train)),2)
	mape_test = round(MAPE(y_test, model.predict(X_test)),2)

	# On met les scores dans des listes
	scores_train = [r2_train, mse_train, mape_train]
	scores_test = [r2_test, mse_test, mape_test]

	# On appelle la fonction qui affiche les scores
	plot_scores(oob, train=scores_train, test=scores_test) 


# Liste de correspondance des noms de régions en abrégé
short_ticks = ['gdp_cap', 'social', 'health', 'freedom', 'trust', 'generosity',
	'C. & East.\n Europe', 'CIS',
	'East\n Asia', 'Latin Am.\n & Carib.', 'Mid.-East &\n N.-Africa',
	'N.-America\n & ANZ', 'South\n Asia', 'Southeast\n Asia',
	'Sub-Saharan\n Africa', 'Western\n Europe']




short_ticks2 = {'gdp_cap':'GDP\nper\nCapita', 'social':'Social', 'health':'Health', 'freedom':'Freedom', 'trust':'Trust', 'generosity':'Generosity',
	'Central and Eastern Europe' : 'Central\n& East.\n Europe', 
	'Commonwealth of Independent States' : 'CIS',
	'East Asia' : 'East\n Asia', 
	'Latin America and Caribbean' : 'Latin\nAmerica\n& Caribbean', 
	'Middle East and North Africa' : 'Mid.-East\n& North-\nAfrica',
	'North America and ANZ' : 'North-\nAmerica\n & ANZ', 
	'South Asia' : 'South\n Asia', 
	'Southeast Asia' : 'South\nEast\nAsia',
	'Sub-Saharan Africa' : 'Sub-\nSaharan\n Africa', 
	'Western Europe' : 'Western\n Europe'}


# Fonction pour afficher les coefficients
def plot_coefs(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, std=[]):
	if hasattr(model, 'coef_'):
		coef = model.coef_
	elif hasattr(model, 'feature_importances_'):
		coef = model.feature_importances_

	# Calcul de l'échelle
	nb = len(coef)
	max_y = round(coef.max() + ((coef.max() - coef.min()) / 3), 1)
	min_y = round(coef.min() - ((coef.max() - coef.min()) / 3), 1)

	# Préparation du graph
	fig = plt.figure(figsize=(16,5))
	ax = fig.add_subplot(111)

	if std != []:
		ax.fill_between(np.arange(len(coef)), coef - std, coef + std, alpha=0.1, color='#186E8B')

	ax.plot(coef, linewidth=4, color='#186E8B')
	ax.plot(coef, 'o', markersize=14, color='#EA0B46')
	ax.plot(coef, 'o', markersize=7, color='white')
	plt.ylim([min_y,max_y])
	plt.grid(alpha=.5)

	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["top"].set_visible(False)


	var_names = []

	for name in X_train:
		var_names.append(short_ticks2[name])

	plt.xticks(range(nb), var_names)

	if min_y < 0 : 
		plt.axhspan(-10, 0, alpha=0.1, color='#F53466')#EA0B46
	plt.show()

	#short_ticks2[X_train.columns.values]


# Fonction pour afficher les erreurs pour 2 régions
@st.cache(hash_funcs={plt.figure: lambda _: None})
def plot_error(model, X_train=X_train, X_test=X_test, list_region=None):

  # Création du dataframe qui récupère les valeurs de y_test et de y_pred pour chaque pays, par région
  results = pd.DataFrame({'region': dfr.loc[dfr['year'] >= 2020, 'region'], 'test': y_test, 'pred': model.predict(X_test)})

  # Instanciation d'une figure et titrage
  fig = plt.figure(figsize=(16,30))  

  # Initialisation du compteur
  count = 1

  if list_region == None:
  	ensemble = results['region'].unique()
  else:
  	ensemble = list_region

  for region in ensemble:
    d = results.loc[results['region'] == region]

    # Calcul des scores
    #r2 = round(r2_score(d['test'], d['pred']),2)
    #mse = round(mean_squared_error(d['test'], d['pred']),2)
    mape = round(MAPE(d['test'], d['pred']),2)
    
    # Création d'un subplot
    ax = fig.add_subplot(10,1,count)

    # Formatage du titre avec LaTeX
    text = region.replace(" ", "\ ")
    plt.title(r"$\mathbf{" + text + "}$" + "  (MAPE=%s)" %(mape), fontsize=16)
    
    
    for j in range(len(d)):
      plt.axvline(x=j, linewidth=.4, color='grey', alpha=.4) 

    sns.scatterplot(x=range(d.shape[0]), y=d['test'], alpha=1, s=60, color='#186E8B', edgecolors='w', label='Réel')
    sns.scatterplot(x=range(d.shape[0]), y=d['pred'], alpha=1, s=60, color='#EA0B46', edgecolors='w', label='Prédit')

    sns.lineplot(x=range(d.shape[0]), y=d['test'], alpha=.7, linewidth=1.5, color='#186E8B')
    sns.lineplot(x=range(d.shape[0]), y=d['pred'], alpha=.7, linewidth=1.5, color='#EA0B46')

    plt.ylim([2,8.5])
    plt.ylabel('score')
    plt.xticks([])
    plt.yticks([2,3,4,5,6,7,8])
    plt.legend()#loc='lower right', bbox_to_anchor=(1.07, .735)
    plt.grid(axis='y', alpha=.6)

    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    count += 1;


# Fonction pour afficher la carte des scores


def plot_map2(year=2021):

	world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

	# On se débarasse des petites iles et de l'antarctique
	world = world[(world.pop_est>0) & (world.name!="Antarctica")]

	# Récupération du fichier contenant les scores et les index des pays
	code_pays = pd.read_csv('datas/noms_codes_pays_worldplot.csv')# index de kaggle, noms de geopandas
	code_pays.rename({'Unnamed: 0':'code'}, axis=1, inplace=True)
	code_pays = code_pays.merge(scores, on='country', how='inner')
	

	# Ajout des scores au df 'world'
	world = world.merge(code_pays, left_index=True, right_on='code', how='left')
	world.loc[world['name'] == 'Greenland', str(year)] = world.loc[world['name'] == 'Denmark', str(year)].values[0]

	w = world.reset_index(drop=True)


	fig = plt.figure(figsize=(14,8))
	ax = fig.add_subplot()

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="5%", pad=0.1)

	w.plot(color='gainsboro', ax=ax)

	w.plot(column=str(year),
	            ax=ax, 
	            cax=cax,
	            legend=True,
	            legend_kwds={'label': "Ladder score", 'orientation': "horizontal"},
				missing_kwds= dict(color = "lightgrey"))

	ax.set_xticks([])
	ax.set_yticks([])
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.set_title('Le sentiment de bonheur noté par pays, en {}'.format(year), fontsize=14, color='k');

	st.pyplot(fig)





###########
# Sidebar #
###########

st.sidebar.image('datas/couv-titre.png', width=280)
st.sidebar.markdown('#')
page_selector = st.sidebar.radio(
	"",
	("Introduction - Contexte", "Présentation des données", "Exploration des données", 
		"Modélisation", "À vous de jouer","Conclusion"))


espace_v = '<div style="padding: 190px 5px;"></div>'
st.sidebar.markdown(espace_v, unsafe_allow_html=True)


signature = '<div style="background-color:#dadfe9;padding: 24px 24px 5px;">\
										<p style="color:grey; font-size: 14px;">Machine learning Project<br>BOOTCAMP DATA ANALYST<br>Juillet-Septembre 2021<br>\
										<a style=color:#f53466; href="https://www.linkedin.com/in/axel-poulier-10701a115/">Axel Poulier</a> - \
										<a style=color:#f53466; href="https://www.linkedin.com/in/matthieuestournet/">Matthieu Estournet</a></p>\
									</div>'
st.sidebar.markdown(signature, unsafe_allow_html=True)





###########################
# Introduction - Contexte #
###########################
if page_selector == 'Introduction - Contexte':

	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Introduction - Contexte</b></span>'
	st.markdown(titre, unsafe_allow_html=True)

	st.title("Y a-t-il une recette du bonheur ?")
	st.markdown("###")
	st.image('datas/couv-map.png')
	st.markdown("###")
	

	st.write("Chaque année, **le 20 mars est** ***la Journée Internationale du Bonheur de l’ONU***. C’est l’occasion pour cette organisation de publier un palmarès des pays les plus heureux au monde, sous le nom de ***World Happiness Report***. \
Ce rapport met en avant le résultat du *Gallup World Poll*, sondage effectué sur un échantillon représentatif de la population dans chaque pays, et recense la sensation de bonheur des habitants en leur posant la question suivante :")

	#st.markdown()

	c1, c2, c3 = st.columns((1,5,1))
	question = '<span style="text-align:center;color:#f53466;font-size: 20px;padding: 0px;"><center><b><i>"Imaginez une échelle allant de 0 à 10, sur laquelle<br>la meilleure vie possible est à 10 et la pire à 0.<br>À quel échelon situeriez-vous votre propre vie ?"</i></b></center></span>'					 
	c2.markdown(question, unsafe_allow_html=True)

#<div style="text-align:center">

#<span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Modélisation</b></span>'


	st.write("Le résultat est un score, sur l’échelle de Cantril, qui permet de classer les pays. Dans ce même rapport, nous trouvons également un certain nombre de données socio-économiques qui permettent de donner un contexte à ce résultat.")

	st.write("Nous allons essayer de voir s’il y a une relation pertinente entre ces indicateurs et le ladder score. Et, si le bonheur est une notion universelle, **est-il possible de trouver une formule pour calculer et prédire la note qu’obtiendra un pays ?** \
C’est ce que nous allons tenter de découvrir au travers de cette étude.")





############################
# Présentation des données #
############################
if page_selector == 'Présentation des données':

	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Présentation des données</b></span>'
	st.markdown(titre, unsafe_allow_html=True)

	st.markdown('### Des datasets provenant de Kaggle')
	st.write("Afin de mener à bien notre étude, nous avons un ensemble de fichiers contenant les scores de bonheur par pays, ainsi que des indicateurs socio-économiques pour les années de 2015 à 2021.")

	st.caption("Voici un extrait du fichier csv pour 2021 (préparé) avec les 5 meilleurs scores :")
	df_max = df.loc[df['year'] == 2021].sort_values(by='score', axis=0, ascending=False)
	st.dataframe(df_max.iloc[:5,:].style.highlight_max('score',axis=0, color="#FA9EB6"))

	st.caption("Voici un extrait du fichier csv pour 2021 (préparé) avec les 5 moins bons scores :")
	df_min = df.loc[df['year'] == 2021].sort_values(by='score', axis=0, ascending=False)
	st.dataframe(df_min.iloc[-5:,:].style.highlight_min('score',axis=0, color="#FA9EB6"))
	
	st.markdown("###")
	st.markdown('### Variable cible')
	'''
	> La variable que nous allons tenter de prédire au mieux est le __ladder score__  qui se trouve dans la colonne _score_. 
	Il est à noter que nous ne disposons pas des données pour tous les pays. Ceux qui ont des valeurs manquantes sont matérialisés en gris sur la carte.

	'''
	year_selected = st.slider('Sélectionnez une année', min_value=2015, max_value=2021, value=2021)

	plot_map2(year_selected)

	st.markdown(' ')
	st.markdown('### Variables explicatives')
	'''
	> Les variables qui vont nous aider à mettre au point un modèle robuste sont les suivantes :
	> * le PIB par habitant _(gdp_cap)_  
	> * le soutien social _(social)_  
	> * l'espérance de vie en bonne santé _(health)_  
	> * la liberté dans les choix de vie de l'individu _(freedom)_  
	> * la générosité perçue _(generosity)_  
	> * le faible niveau de corruption perçue _(trust)_  

	'''
	
	nb_pays = 149
	nb_regions = 10

	st.markdown(' ')
	st.write('### Nombre de pays :', nb_pays)
	st.write("Nous avons un maximum de 149 pays. Certaines années, des données sont manquantes pour certains pays.")
	st.markdown(' ')
	st.write('### Nombre de régions du globe :', nb_regions)
	st.write("Les pays sont également répartis par régions dans un des 10 groupes suivants :")
	st.write('Central and Eastern Europe, Commonwealth of Independent States, East Asia, Latin America and Caribbean, Middle East and North Africa, North America and ANZ, South Asia, Southeast Asia, Sub-Saharan Africa, Western Europe')
	st.markdown(' ')

	st.write('### Pre-processing')
	st.write("L'ensemble de nos variables explicatives ont été normalisées avec un **MinMaxScaler** afin de ne pas donner plus d'importance à celles qui ont l'échelle la plus grande. De plus, les variables de régions ont été _dichotomisées_. De ce fait toutes ces variables auront une échelle comprise entre 0 et 1. ")
	st.write("Nous allons ensuite séparer nos données en un ensemble d'entraînement et un ensemble de test. Le premier groupe sera composé des données de 2015 à 2019 inclus. Et nous vérifierons nos modèles sur les données de 2020 et 2021.")




###########################
# Exploration des données #
###########################
if page_selector == 'Exploration des données':
	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Exploration des données</b></span>'
	st.markdown(titre, unsafe_allow_html=True)

	# Distribution des variables
	st.markdown('### Distribution de la variable cible')
	st.write("Nous pouvons constater grâce au graphique ci-dessous que la distribution des scores est assez symétrique et étendue. La plupart des valeurs se trouve entre 4 et 6,5. Il est à noter également qu'il y a une petite remontée sur les valeurs hautes (>7).")

	fig = plt.figure(figsize=(12,2))
	
	ax = fig.add_subplot()
	ax.set_yticks([])
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["top"].set_visible(False)
	#hue = ['#186E8B' if (x < max(values)) else 'red' for x in values ]
	sns.histplot(df['score'], color='#186E8B', bins=17, edgecolor='white');#, palette=clrs

	st.pyplot(fig)

	st.markdown("###")

	# Distribution des variables
	st.markdown('### Distribution des variables explicatives')
	c1, c2, c3, c4, c5 = st.columns((8,1,8,1,8))
	
	t1 = "<p style='text-align:justify;font-size: 14px'>Le PIB par habitant en parité de pouvoir d'achat est distribué de manière relativement harmonieuse, avec un effectif plus important dans la moitié supérieure.</p>"
	
	c1.write(t1, unsafe_allow_html=True)
	t2 = "<p style='text-align:justify;font-size: 14px'>À la question \"Avez-vous confiance en vos institutions et dans les entreprises de votre pays\", il y a une majorité de réponses positives. Et nous pouvons noter une césure au milieu, ce qui tend à démontrer qu'au sein d'un même pays il y a rarement un équilibre des réponses.</p>"
	c3.write(t2, unsafe_allow_html=True)
	t3 = "<p style='text-align:justify;font-size: 14px'>Ce graphique montre que la grande majorité des personnes interrogées dans chaque pays a répondu ne pas avoir donné à des oeuvres au cours des derniers mois. .</p>"
	c5.write(t3, unsafe_allow_html=True)
	


	fig = plt.figure(figsize=(12,2))
	

	for i, var in enumerate (df[['gdp_cap','trust','generosity']]):
		ax = fig.add_subplot(1,3,i+1)
		ax.set_yticks([])
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		ax.spines["top"].set_visible(False)
		sns.histplot(df[var], bins=10, color='#186E8B', edgecolor='white');

	st.pyplot(fig)


	
	
	#st.write("---")
	t4 = "<p style='text-align:justify;font-size: 14px'>Les 3 indicateurs suivants ont la même tendance. La majorité des pays ont un score qui se situe entre 0,6 et 0,9. Ce qui indique que l'auto-détermination, le tissu social et l'espérance de vie en bonne santé sont plutôt élevés.</p>"
	st.write(t4, unsafe_allow_html=True)
	

	fig = plt.figure(figsize=(12,2))
	#fig.suptitle('Distribution des variables', fontsize=14)

	for i, var in enumerate (df[['freedom','social','health']]):
		ax = fig.add_subplot(1,3,i+1)
		ax.set_yticks([])
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		ax.spines["top"].set_visible(False)
		sns.histplot(df[var], bins=10, color='#186E8B', edgecolor='white');

	st.pyplot(fig)

	st.markdown("###")

	# HEATMAP
	st.markdown('### Matrice des corrélations')


	st.write("La heatmap suivante nous montre à quel point les variables sont correlées entre elles. Nous observons tout particulièrement le lien entre la variable cible et les variables *gdp_cap*, social_ et _health_. Et dans une moindre mesure avec _freedom_. À l'inverse, _trust_ et _generosity_ semblent n'avoir que peu d'influence sur la variable cible.")

	corr = df.iloc[:,:9].corr()
	fig = plt.figure(figsize=(12,4))
	ax = fig.subplots()
	g = sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', annot_kws={"size": 12})
	
	st.pyplot(fig)

	st.markdown("###")

	# Vs graphs
	st.markdown('### Corrélation entre le score et les variables explicatives')
	st.write("Ci-dessous, vous pouvez observer plus en détail le lien qu'il peut y avoir entre les variables explicatives. Pour commencer, gdp_cap Vs trust : il est très clair sur cette représentation des données, que la variable *gdp_cap* est très corrélée à la variable cible _score_. En revanche, avec la variable _trust_, ce n'est pas du tout le même aspect. La droite de régression est presque horizontale, ce qui montre une faible corrélation. Et les points sont amalgamés en 2 ensembles. Ce qui rejoint l'observation de la distribution de cette variable, plus haut sur cette page.")
	col1, col2 = st.columns(2)

	# Colonne 1
	select1 = col1.selectbox(' ', df.columns[3:9].tolist(), index=0)
	vs_score_plot(x=df[select1], title='Score en fonction de '+select1, col=col1)

	# Colonne 2
	select2 = col2.selectbox(' ', df.columns[3:9].tolist(), index=4)
	vs_score_plot(x=df[select2], title='Score en fonction de '+select2, col=col2)







################
# Modélisation #
################
if page_selector == 'Modélisation':

	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Modélisation</b></span>'
	st.markdown(titre, unsafe_allow_html=True)


	#################### >> Régression linéaire << ####################

	st.subheader('Régression linéaire')
	
	st.write("Le premier modèle que nous allons entraîner est une régression linéaire. Cet algorithme est facile à mettre en place et à évaluer. Il va nous servir de référence pour ensuite pouvoir utiliser d'autres modèles et faire évoluer la précision de nos prédictions.")
	st.write("Pour ce premier modèle, nous utilisons les paramètres par défaut, ainsi que l'ensemble de nos variables explicatives.")
	st.write(" ")

	st.markdown('#### Évaluation de la régression linéaire')
	
	st.pyplot(print_perf(model=lr))
	st.write("Notre modèle semble relativement performant. Le coefficient de détermination (R2) est élevé. Mais la différence entre la note sur l'ensemble de test et sur l'ensemble d'entraînement suggère qu'il y a du sur-apprentissage. Ceci est confirmé par l'erreur quadratique moyenne (MSE) qui n'est pas convenable avec un écart assez important en défaveur de l'ensemble de test. Quant à la MAPE qui nous indique la moyenne d'erreur absolue en pourcentage, elle est assez forte et doit être améliorée.")
	st.write(" ")

	st.markdown('#### Coefficients utilisés pour chaque variable')
	st.pyplot(plot_coefs(model=lr))
	st.write("Comme nous pouvions nous y attendre, les coefficients appliqués à chaque variable sont plus grands pour nos 6 premières variables explicatives principales (i.e. hors régions). Cependant, il est aussi à noter que les coefficients appliqués aux régions peuvent être assez disparates.")
	
	st.write("#")


	#################### >> RFE << ####################

	st.subheader('Recursive Feature Elimination (RFE)')

	st.write("Nous allons vérifier l’importance des variables grâce à une RFE sur la régression linéaire. Ceci va nous permettre de sélectionner les variables explicatives les plus importantes et probablement de réduire l'overfitting.")
	st.write(" ")

	st.markdown('#### Évaluation de la RFE')
	st.pyplot(print_perf(model=rfe))
	#X_train.columns[rfe.support_].values
	st.write("D'après la courbe précédente, nous avons 4 variables qui se détachent au-dessus de 0,6 en coefficients. La RFE confirme que les 4 variables les plus importantes sont bien 'gdp_cap', 'social', 'health' et 'freedom'. Et avec ces 4 variables, nous obtenons des scores plus serrés. Le sur-apprentissage a été sensiblement réduit. Mais la performance du modèle n’est pas totalement satisfaisante.")
	st.write("Pour essayer de créer un modèle plus performant nous allons ensuite nous intéresser à Ridge.")

	st.write("#")

	#################### >> Ridge << ####################

	st.subheader('Ridge')
	st.write("Ridge va nous permettre d’ajouter une contrainte (alpha) sur les coefficients lors de la modélisation avec pour objectif de réduire les erreurs.")
	st.write(" ")

	# on affiche les résultats obtenus
	st.markdown('#### Évaluation du modèle Ridge')
	st.pyplot(print_perf(model=ridge2, 
		X_train=X_train.iloc[:,:4], y_train=y_train, 
		X_test=X_test.iloc[:,0:4], y_test=y_test))

	st.write("L’alpha (la contrainte) qui a eu le meilleur score est 1.")
	st.write("Tous nos scores sur train et sur test sont identiques à ceux de la RFE.")
	st.write(" ")
	
	st.markdown('#### Coefficients utilisés pour chaque variable')
	st.pyplot(plot_coefs(model=ridge2, 
		X_train=X_train.iloc[:,:4], y_train=y_train, 
		X_test=X_test.iloc[:,0:4], y_test=y_test))
	st.write("Ici, nous observons les coefficients utilisés sur les 4 premières variables.")
	st.write("La mise en place d'une pénalisation L2 n'a pas permis d'améliorer le modèle. Nous allons voir si une pénalisation L1 (Lasso) donnera d'autres résultats.")
	st.write("#")

	#################### >> Lasso << ####################

	st.subheader('Lasso')
	st.write("Le modèle lasso permet de supprimer des variables en mettant leur poids à zéro. C'est le cas si deux variables sont corrélées. L’une sera conservée, l’autre non. C’est un modèle qui fait une sélection de variables.")
	st.write(" ")

	st.markdown('#### Évaluation du modèle Lasso')
	st.pyplot(print_perf(model=lasso_cv))
	st.write(" ")
	st.write(" ")

	st.markdown('#### Coefficients utilisés pour chaque variable')
	st.pyplot(plot_coefs(model=lasso_cv))
	st.write(" ")
	st.write("#")

#################### >> Random Forest default << ####################

	st.subheader('Random Forest #1')
	st.markdown('Paramètres par défaut')
	st.write("La forêt aléatoire est un modèle qui construit un ensemble d’arbres de décision, généralement formés à la méthode de \"mise en sac\" (bootstrap). L'idée générale de la méthode d'ensachage est qu'une combinaison de modèles d'apprentissage augmente le résultat global. Le bootstrap crée des échantillons sur lesquels le modèle va s’entrainer. Ces échantillons sont formés à partir des données tirées au hasard dans l’ensemble d’entraînement, avec remise. Ils peuvent donc être tirés plusieurs fois pour constituer l’échantillon. Mais certaines données ne seront pas tirées au hasard. Elles serviront alors d’échantillons de validation dans le oob score (out of bag).")
	st.write(" ")

	st.markdown('#### Évaluation de la random forest #1')
	st.pyplot(print_perf(model=rf1))
	st.write("Avec les hyperparamètres de base, ce modèle est plus performant que ce que nous avons obtenu jusqu’à maintenant. Néanmoins, nous constatons aussi qu’il y a une différence importante entre les scores sur train et sur test. Vu les faibles MSE et MAPE sur train, il y a clairement du sur-apprentissage.")
	st.write(" ")

	

	st.markdown('#### Importance des variables')
	std = np.std([tree.feature_importances_ for tree in rf1.estimators_], axis=0)
	st.pyplot(plot_coefs(model=rf1, std=std))
	st.write("#")

#################### >> Random Forest #2 << ####################
	st.subheader('Random Forest #2')
	st.markdown('max_depth = 3 et uniquement les variables social, health et freedom')

	st.write("Nous allons nous intéresser aux variables les plus importantes. Après plusieurs essais, c'est avec ‘social’, ‘health’ et 'freedom' que nous obtenons le meilleur résultat. Afin de réduire l’overfitting, nous allons également régler le paramètre max_depth à 3.")

	st.markdown('#### Évaluation de la random forest 2')
	st.pyplot(print_perf(model=rf2, X_train=X_train.iloc[:,1:4], X_test=X_test.iloc[:,1:4]))
	st.write("La lutte contre l’overfitting a porté ses fruits. Les résultats sur train et test sont assez équilibrés. En revanche la MSE et la MAPE restent assez élevées. Le pourcentage d'erreur absolue tourne autour de 8%.")

	st.write(" ")



	

	st.markdown('#### Importance des variables')
	std = np.std([tree.feature_importances_ for tree in rf2.estimators_], axis=0)
	st.pyplot(plot_coefs(model=rf2, 
		X_train=X_train.iloc[:,1:4], 
		X_test=X_test.iloc[:,1:4], 
		std=std))
	st.write("#")


#################### >> Random Forest #3 << ####################
	st.subheader('Random Forest #3')
	st.write('max_depth = 6 avec 4 variables')
	st.write("Afin d'améliorer encore notre modèle, nous avons utilisé un GridSearchCV sur la Random Forest. Les meilleurs paramètres qui ont été retenus sont les suivants : max_depth=6 et utilisation des 4 premières variables, à savoir 'gdp_cap', 'social', 'health' et 'freedom'.")
	
	st.markdown('#### Évaluation de la random forest 3')
	st.pyplot(print_perf(model=rf3, X_train=X_train.iloc[:,:4], X_test=X_test.iloc[:,:4]))#
	st.write("Nous avons réussi à améliorer la performance du modèle. L'écart entre les échantillons d'entraînement et de test est meilleur et ne montre pas un sur-apprentissage manifeste. Les prédictions sont également plus précises qu'auparavant.")
	st.write(" ")

	st.markdown('#### Importance des variables')
	std = np.std([tree.feature_importances_ for tree in rf3.estimators_], axis=0)
	st.pyplot(plot_coefs(model=rf3, 
		X_train=X_train.iloc[:,:4], 
		X_test=X_test.iloc[:,:4], 
		std=std))
	st.write("Les variables utilisées n'ont pas toutes la même importance. C'est 'health' qui est la plus discriminante, comme on le voit avec les modèles précédents également.")
	st.write("Par la suite, nous pouvons observer la répartition des erreurs selon la région du globe avec le graphique ci-dessous.")
	st.write(" ")

	st.markdown('####	 Scores prédits et scores réels, par région')
	st.markdown('Ensemble de test, années 2020 et 2021')


	st.pyplot(plot_error(model=rf3, X_train=X_train.iloc[:,:4], X_test=X_test.iloc[:,:4]))#







###################
# À vous de jouer #
###################
if page_selector == 'À vous de jouer':

	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>À vous de jouer</b></span>'
	st.markdown(titre, unsafe_allow_html=True)

	st.write("Vous pensez pouvoir trouver de meilleures prédictions ? Alors lancez-vous dans le tuning sur un des modèles suivants !!…")
	st.write("")

	model_selector = st.selectbox("Choix du modèle", ['Regression Linéaire','Ridge','Random Forest'])

	var_list = X_train.columns.values.tolist()
	
	var_selected = st.multiselect("Choix des variables", var_list, var_list)	
	

	if model_selector == 'Regression Linéaire':
		validate = st.button('Évaluer le modèle')

		if validate:
			model = LinearRegression()
			XTR = X_train[var_selected]
			XTE = X_test[var_selected]
			model.fit(XTR, y_train)

			# Graph évaluation du modèle
			st.markdown('#### Évaluation du modèle '+ model_selector)
			st.pyplot(print_perf(model=model, X_train=XTR, X_test=XTE))

			# Graph coefficients
			st.markdown('#### Importance des variables')
			st.pyplot(plot_coefs(model=model, X_train=XTR, X_test=XTE))

			# Graph erreurs
			st.markdown('####	 Scores prédits et scores réels, par région')
			st.pyplot(plot_error(model=model, X_train=XTR, X_test=XTE))


	if model_selector == 'Ridge':
		alpha = st.number_input('Alpha (>0) pour le modèle ' + model_selector)
		validate = st.button('Évaluer le modèle')

		if validate:
			model = Ridge(alpha=alpha)
			XTR = X_train[var_selected]
			XTE = X_test[var_selected]
			model.fit(XTR, y_train)

			# Graph évaluation du modèle
			st.markdown('#### Évaluation du modèle '+ model_selector)
			st.pyplot(print_perf(model=model, X_train=XTR, X_test=XTE))

			# Graph coefficients
			st.markdown('#### Importance des variables')
			st.pyplot(plot_coefs(model=model, X_train=XTR, X_test=XTE))

			# Graph erreurs
			st.markdown('####	 Scores prédits et scores réels, par région')
			st.pyplot(plot_error(model=model, X_train=XTR, X_test=XTE))


	if model_selector == 'Random Forest':
		#num_trees = st.number_input("Nombre d'arbres de la " + model_selector, value=100)
		num_trees = st.slider("Nombre d'arbres", min_value=1, max_value=500, value=100, step=1)
		depth = st.slider("Profondeur max.", min_value=1, max_value=100, value=10, step=1)
		
		validate = st.button('Évaluer le modèle')

		
		
		if validate:
			model = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, n_jobs=-1, oob_score=True)
			XTR = X_train[var_selected]
			XTE = X_test[var_selected]
			model.fit(XTR, y_train)

			# Graph évaluation du modèle
			st.markdown('#### Évaluation du modèle '+ model_selector)
			st.pyplot(print_perf(model=model, X_train=XTR, X_test=XTE))

			# Graph coefficients
			st.markdown('#### Importance des variables')
			st.pyplot(plot_coefs(model=model, X_train=XTR, X_test=XTE))

			# Graph erreurs
			st.markdown('####	 Scores prédits et scores réels, par région')
			st.pyplot(plot_error(model=model, X_train=XTR, X_test=XTE))



	
	

	





##############
# Conclusion #
##############
if page_selector == 'Conclusion':

	titre = '<span style="color:black;font-size: 14px;padding: 0px;">Happiness Recipy<br></span>\
					 <span style="color:#f53466;font-size: 20px;padding: 0px;"><b>Conclusion</b></span>'
	st.markdown(titre, unsafe_allow_html=True)


	tx = '<p style="color:black;text-align:justify;font-size: 16px;padding: 0px;">Afin de prédire le ladder score des pays, nous avons essayé plusieurs modèles de régression avec différents paramètres et différentes combinaisons de variables. Et, bien que nous ayons fait progresser notre modèle de façon claire, le taux d’erreurs dans nos prédictions reste à un niveau qui n\'est pas satisfaisant.<br>\
	<br>\
	Cependant nous avons pu nous rendre compte que la différence entre scores prédits et scores réels a l\'air d\'être en lien avec la région du globe. Comme nous venons de le voir, des régions comme Sub-Saharan Africa ou SouthEast Asia n’ont pas des résultats très performants. Pourtant le même modèle a de biens meilleurs résultats sur North America and ANZ ou Western Europe.<br>\
	<br>\
	La recette qui constitue le bonheur est différente pour chacun de nous et encore davantage pour chaque culture de chaque pays ou région. C’est pourquoi, créer un seul modèle pour prédire de la même façon le ladder score aux États-Unis ou au Zimbabwe paraît assez difficile. L’importance des choses matérielles n’est par exemple pas la même dans nos sociétés occidentales que dans beaucoup de pays de la zone intertropicale. Il en va de même pour l’importance du tissu social.<br>\
	<br>\
	Il pourrait être intéressant d’essayer de créer un modèle spécifique à chaque région, qui prendrait éventuellement en compte d’autres variables. Nous pouvons aussi imaginer créer certains regroupements. Par exemple North America and ANZ pourrait être réunis avec Western Europe en tant que pays occidentaux.</p>'
	st.markdown(tx, unsafe_allow_html=True)
	
	st.write("##")

	caption = '<p style="color:grey;font-size: 14px;padding: 0px;">Si vous avez apprecié notre travail ou si vous avez des questions, des propositions, n\'hésitez pas à nous contacter grâce aux liens sur la gauche de l\'écran.</p>'
	st.markdown(caption, unsafe_allow_html=True)

	st.image('datas/couverture-rapport.jpg')










