from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


# Carregar os dados dos arquivos CSV
buy = pd.read_csv('base/Buy.csv')
abt = pd.read_csv('base/Abt.csv', encoding_errors='ignore')
goalDf = pd.read_csv('base/abt_buy_perfectMapping.csv')

vectorizer = TfidfVectorizer()

# Remover colunas desnecessárias
buy = buy.drop(columns=[ 'manufacturer', 'price'])
abt = abt.drop(columns=['price'])

# Substituir valores nulos nas colunas de texto por uma string vazia
buy.fillna({'name': '', 'description': ''}, inplace=True)
abt.fillna({'name': '', 'description': ''}, inplace=True)


combined = pd.concat([buy, abt], ignore_index=True)
combined = combined.sort_values(by=['name'])
clusters = 450
kmeans = KMeans(n_clusters=clusters, max_iter=20000, algorithm='elkan')
matrixc = vectorizer.fit_transform(combined['name'] + ' ' + combined['description'])
kmeans.fit(matrixc)
combined['Cluster'] = kmeans.labels_
print(combined[:5])
print(combined.loc[combined['id'] == 10011646])
print(combined.loc[combined['id'] == 38477])

# get the all the items that are in the same cluster
duplicatesTrue = []
duplicatesFalse= []
for i in range(clusters):

    items = combined.loc[combined['Cluster'] == i]
    
    #print(items)
    if(items.__len__() > 1):
        print('cluster:',i)
        items_ids = items['id'].to_list()
        for j in range(len(goalDf)):
            if(goalDf.iloc[j]['idAbt'] in items_ids and goalDf.iloc[j]['idBuy'] in items_ids):
                duplicatesTrue.append(items)
                print(items_ids)
                print(goalDf.iloc[j])
            
        duplicatesFalse.append(items)

lenTrue = len(duplicatesTrue)
lenFalse = len(duplicatesFalse)
lenGoal = 1098
print(len(duplicatesTrue), len(duplicatesFalse))
print('total', lenTrue+lenFalse)
print('%', lenTrue/lenGoal*100)


exit()
