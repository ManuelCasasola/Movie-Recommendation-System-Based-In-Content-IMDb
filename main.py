import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


# Tras haber descargado el top 1000 peliculas de la página IMDB
# Procedemos a cargar el Dataset.
imdb=pd.read_csv('imdb_top_1000.csv')
imdb.head()


#Filtramos los datos
cols=['Genre','Director','Star1','Star2','IMDB_Rating', 'Meta_score']
filtro_imdb=imdb.loc[:,cols]


#Generamos una canalizacion para variables númericas
canalizacion_numerica=Pipeline([
    ('scaler',StandardScaler())
    ])

#Generamos una canalizacion para variables categóricas
cols_categoricas=['Genre','Director','Star1','Star2']
canalizacion_categorica=Pipeline([
    ('encoder',OneHotEncoder(drop='first'))
    ])
canalizacion_categorica.fit(filtro_imdb[cols_categoricas])
filtro_imdb_encoded = canalizacion_categorica.transform(filtro_imdb[cols_categoricas])


#Vamos a calcular la matriz de similitudes entre las diferentes películas
n_neighbors=5
nneighbors=NearestNeighbors(n_neighbors=n_neighbors,metric='cosine').fit(filtro_imdb_encoded)

dif, ind = nneighbors.kneighbors(filtro_imdb_encoded[2])

print("Película que te gusta")
print("="*80)
print(imdb.loc[ind[0][0], :])
print("Películas recomendadas")
print("="*80)
print(imdb.loc[ind[0][1:], :])