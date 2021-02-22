import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1 
dataset = pd.read_csv(r'C:\Users\Pichau\Downloads\04_dados_aula.csv')

#2
features = dataset.iloc[:,:-1].values
classe = dataset.iloc[:,-1].values

#3
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

#4
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
features = np.array(columnTransformer.fit_transform(features))

#5
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)

#6
from sklearn.model_selection import train_test_split
features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(features,classe,test_size=0.15,random_state=1)

#7
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
features_treinamento[:,3:] = standardScaler.fit_transform(features_treinamento[:,3:])
features_teste[:,3:] = standardScaler.transform(features_teste[:,3:])