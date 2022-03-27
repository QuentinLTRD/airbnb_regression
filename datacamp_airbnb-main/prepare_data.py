import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv(r"data\airbnb_bordeaux.csv")

# Nettoyage du dataset
# On prend le parti de considérer "strictes" les conditions d'annulation non renseignées
df = df.fillna({"conditions_annulation": "Strictes"})
df = df.replace(to_replace={'Flexibles': 'flexibles', 'Modérées': 'moderees', 'Strictes': 'strictes'})
# On effectue un one hot encoding sur la variable conditions_annulation
df = pd.get_dummies(df, columns=["conditions_annulation"])

# Par simplicité, on ne prend en compte que le fait que les animaux soient autorisés ou pas
# (car le type d'animal risque de créer du bruit de manière inutile pour prédire le prix)
# On remplace par des 0 quand pas d'animal autorisé, 1 quand autorisé
df["animal_sur_place"] = df["animal_sur_place"].notna().astype('int')

# On crée une catégorie "multiples" pour les types de lits non renseignés
# (car ils correspondent à plusieurs types de lits à la fois pour le même logement)
df = df.fillna({"type_lit": "multiples"})
df = df.replace(to_replace={'Vrai lit': 'vrai_lit', 'Canapé convertible': 'canape_convertible', 'Canapé': 'canape'})
df = pd.get_dummies(df, columns=["type_lit"])

# On regroupe les types de propriétés similaires
df = df.replace(to_replace={'Appartement': 'appartement', 'Maison': 'maison', 'Maison de ville': 'maison_de_ville',
                            'Bed & Breakfast': 'bed_and_breakfast', 'Appartement en résidence': 'appart_en_residence',
                            'Loft': 'loft', 'Inconnue': 'inconnu', 'Autre': 'autre',
                            'Bungalow': 'bungalow_cabane_dortoir_eco', 'Villa': 'maison',
                            'Cabane': 'bungalow_cabane_dortoir_eco', 'Maison écologique': 'bungalow_cabane_dortoir_eco',
                            'Dortoir': 'bungalow_cabane_dortoir_eco'})
df = pd.get_dummies(df, columns=["type_propriete"])

df = df.replace(to_replace={'Logement entier': 'logement_entier', 'Chambre privée': 'chambre_privee',
                            'Chambre partagée': 'chambre_partagee'})
df = pd.get_dummies(df, columns=["type_logement"])

# On retire les lignes dont le prix est nul
df = df.drop(df[df["prix_nuitee"] == 0].index)

# On modifie les lignes pour lesquelles le nombre de salles de bain est extravagant et ne correspond pas au vrai nombre
# indiqué dans l'annonce
df.loc[df['nombresdb'] >= 15] = 1

# On corrige les fautes de frappes
df = df.rename(columns={'rection_semaine': 'reduction_semaine'})

# On retire la colonne shampooing (qui vaut 0 partout) et les colonnes contenant des chaînes de caractères
# On retire aussi la colonne "prixnuitee" présente deux fois (on ne garde que "prix_nuitee")
df = df.drop(columns=['reglement_interieur', 'description', 'resume', 'titre',
                      'url', 'identifiant', 'shampooing', 'prixnuitee'])

# On veut stratifier équitablement selon les prix des nuitées pour diviser en dataset public et privé
df["y_binned"] = pd.cut(df["prix_nuitee"], bins=3, labels=np.arange(3), right=False)
df_public, df_private = train_test_split(df, shuffle=True, train_size=0.7,
                                         stratify=df["y_binned"], random_state=26)

# On divise chaque dataset en train et test de manière stratifiée sur le prix à la nuit
df_public_train, df_public_test = train_test_split(df_public, shuffle=True, train_size=0.8,
                                                   stratify=df_public["y_binned"], random_state=26)

df_private_train, df_private_test = train_test_split(df_private, shuffle=True, train_size=0.8,
                                                     stratify=df_private["y_binned"], random_state=26)

# On retire la colonne "y_binned" ajoutée pour stratifier
df_private_train = df_private_train.drop(columns=["y_binned"])
df_private_test = df_private_test.drop(columns=["y_binned"])
df_public_train = df_public_train.drop(columns=["y_binned"])
df_public_test = df_public_test.drop(columns=["y_binned"])

# On enregistre les données dans des fichiers
df_private_train.to_csv(os.path.join('data', 'private_train.csv'), index=False)
df_private_test.to_csv(os.path.join('data', 'private_test.csv'), index=False)
df_public_train.to_csv(os.path.join('data', 'train', 'train.csv'), index=False)
df_public_test.to_csv(os.path.join('data', 'test', 'test.csv'), index=False)
