# Projet CentraleSupélec de prédiction de consommation énergétique
Ce projet a été effectué par Paul Castéras, Aymeric Ducatez, Hugo Heitz et Arthur Walker dans le cadre de la mention "Mathématiques et Data Science" de l'École CentraleSupélec.

Tous les fichiers peuvent être lancés séparément, et correspondent à différentes méthodes de prédiction.
- `exploration.ipynb` constitue une bonne introduction au projet.
- `elasticNet_avec_autoencodeur.py` et `elasticNet_sans_autoencodeur.py` représentent la méthode de la toile élastique appliquée à la prédiction de la consommation d'un client en utilisant la consommation des clients témoins au même instant.

Le code dans le dossier "alphacsc" a été repris puis modifié à partir de répertoire : https://github.com/alphacsc/alphacsc.git
Trois fichiers ont été ajouté dans ce répertoire :
- `build_dictionnary.py` construit l'approximation du signal
- Les deux notebooks portent sur des méthodes de régression à partir des dictionnaires déjà construit. Il faut exécuter `build_dictionnary.py` avant les notebook pour construire les signaux de références et les signaux d'activations.

Le code dans le dossier "patchtst" correspond à l'utilisation du modèle PatchTST par la librairie `neuralforecast`.
Il est constitué de deux fichiers :
- `data_preprocessing.py` construit le fichier de données qui sera utilisé par le modèle
- Le notebook `patchtst.ipynb` importe ces données, les fournit au modèle et affiche les résultats des prédictions.
