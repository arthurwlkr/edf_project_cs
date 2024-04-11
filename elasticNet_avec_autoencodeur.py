import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import Timestamp
from sklearn.linear_model import ElasticNet
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Chargement des données
calendrier_path = "Données/challenge_data/calendrier_challenge.parquet"
questionnaire_path = "Données/challenge_data/questionnaire.parquet"
consos_challenge_path = "Données/challenge_data/consos_challenge.parquet"

calendrier = pd.read_parquet(calendrier_path)
questionnaire = pd.read_parquet(questionnaire_path)
consos_challenge = pd.read_parquet(consos_challenge_path)

# Nettoyage
questionnaire = questionnaire[["id_client", "participe_challenge"]]
calendrier = calendrier[["horodate"]]

# On récupère l'info de la participation au challenge
merged_data = pd.merge(consos_challenge, questionnaire, on='id_client')

# On sépare en entrée et sortie
challenge_data = merged_data[merged_data['participe_challenge']]
X = pd.pivot_table(challenge_data, values='puissance_W', index='horodate', columns='id_client').fillna(0)
non_challenge_data = merged_data[~merged_data['participe_challenge']]
Y = pd.pivot_table(non_challenge_data, values='puissance_W', index='horodate', columns='id_client').fillna(0)

X = X.reset_index()
Y = Y.reset_index()

# On sépare nos données en données d'entraînement, de validation et de test.
X_test = X[X["horodate"].isin(calendrier["horodate"])]
X_nontest = X[~X["horodate"].isin(calendrier["horodate"])]
Y_nontest = Y[~Y["horodate"].isin(calendrier["horodate"])]

dates = sorted(merged_data['horodate'].unique())
train_dates = [d for d in dates if (d < Timestamp('2010-07-15 00:00:00'))]
validation_dates = [d for d in dates if (d >= Timestamp('2010-07-15 00:00:00'))]


X_train = X_nontest[X_nontest["horodate"].isin(train_dates)]
Y_train = Y_nontest[Y_nontest["horodate"].isin(train_dates)]


X_train = X_train.set_index("horodate")
Y_train = Y_train.set_index("horodate")

# Extraction de statistiques
X_mean = np.mean(X_train, axis=1)
X_median = np.median(X_train, axis=1)
X_min = np.min(X_train, axis=1)
X_max = np.max(X_train, axis=1)
X_std = np.std(X_train, axis=1)

# Normalisation des données d'entrée
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Définition de l'auto-encodeur
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(307, 128)
        self.decoder = nn.Linear(128, 307)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

# Création du modèle
autoencoder = Autoencoder()

# Fonction de perte et optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Entraînement de l'auto-encodeur
for epoch in range(10):
    print("epoch :",epoch)
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    outputs = autoencoder(inputs)
    loss = criterion(outputs, inputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extraction des caractéristiques de la couche cachée
with torch.no_grad():
    encoded = autoencoder.encoder(torch.tensor(X_train_scaled, dtype=torch.float32))
    X_encoded = encoded.numpy()

# Concaténation des caractéristiques avec les statistiques
X_features = np.column_stack((X_mean, X_median, X_min, X_max, X_std, X_encoded))

alpha = 0.2
l1_ratio = 0.1


X_validation = X[X["horodate"].isin(validation_dates)]
X_validation = X_validation.set_index("horodate")


# Entraînement de l'ElasticNet
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
print("Entraînement de l'Elastic Net ...")
elastic_net.fit(X_features, Y_train)


X_validation_scaled = scaler.transform(X_validation)
X_mean_val = np.mean(X_validation_scaled, axis=1)
X_median_val = np.median(X_validation_scaled, axis=1)
X_min_val = np.min(X_validation_scaled, axis=1)
X_max_val = np.max(X_validation_scaled, axis=1)
X_std_val = np.std(X_validation_scaled, axis=1)
with torch.no_grad():
    encoded_val = autoencoder.encoder(torch.tensor(X_validation_scaled, dtype=torch.float32)).numpy()
X_features_val = np.column_stack((X_mean_val, X_median_val, X_min_val, X_max_val, X_std_val, encoded_val))

# Prédiction sur les données de validation
Y_pred_val = elastic_net.predict(X_features_val)

Y_pred_val_df = pd.DataFrame(Y_pred_val, columns=Y_validation.columns.drop("horodate"))

Y_validation = Y[Y["horodate"].isin(validation_dates)]

# On ajoute les dates :
Y_validation = Y_validation.reset_index(drop=True)
Y_pred_val_df = Y_pred_val_df.reset_index(drop=True)
Y_pred_val_df["horodate"] = Y_validation["horodate"]

Y_pred_val_8_13 = Y_pred_val_df[(Y_pred_val_df["horodate"].dt.hour >= 8) & (Y_pred_val_df["horodate"].dt.hour <= 13)]
Y_validation_8_13 = Y_validation[(Y_validation["horodate"].dt.hour >= 8) & (Y_validation["horodate"].dt.hour <= 13)]

Y_pred_val_18_21 = Y_pred_val_df[(Y_pred_val_df["horodate"].dt.hour >= 18) & (Y_pred_val_df["horodate"].dt.hour <= 21)]
Y_validation_18_21 = Y_validation[(Y_validation["horodate"].dt.hour >= 18) & (Y_validation["horodate"].dt.hour <= 21)]

Y_validation = Y_validation.drop('horodate',axis = 1)
Y_pred_val_df = Y_pred_val_df.drop('horodate',axis = 1)
Y_validation_8_13 = Y_validation_8_13.drop('horodate',axis = 1)
Y_pred_val_8_13 = Y_pred_val_8_13.drop('horodate',axis = 1)
Y_validation_18_21 = Y_validation_18_21.drop('horodate',axis = 1)
Y_pred_val_18_21 = Y_pred_val_18_21.drop('horodate',axis = 1)

# Calcul de la MSE sur la période 8h-13h :
mse_8_13 = np.mean((Y_pred_val_8_13 - Y_validation_8_13) ** 2)

# Calcul de la MSE sur la période 18h-21h :
mse_18_21 = np.mean((Y_pred_val_18_21 - Y_validation_18_21) ** 2)

# Calcul de la MSE totale :
mse_totale = np.mean((Y_pred_val_df - Y_validation) ** 2)

print("MSE sur 8-13 : ", mse_8_13)
print("MSE sur 18-21 : ", mse_18_21)
print("MSE totale : ", mse_totale)

# Représentation d'un client sur une journée


X_time = np.linspace(0,23.5,48)
Y_pred_val_1005 = Y_pred_val_df[1047].values[:48]
Y_validation_1005 = Y_validation[1047].values[:48]

plt.plot(X_time, Y_pred_val_1005, label='Consommation estimée')
plt.plot(X_time, Y_validation_1005, label='Consommation réelle')
plt.xticks(rotation=45)
plt.title('Consommation estimée vs Consommation réelle')
plt.xlabel('Time of day (h)')
plt.ylabel('Consumption')

plt.legend()

plt.show()