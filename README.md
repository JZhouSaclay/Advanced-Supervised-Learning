# Advanced-Supervised-Learning

1. - # Hotel Booking Cancellation Prediction - Classification Task

   ## 1. Project Overview

   This is a multi-class classification machine learning task, aiming to predict the final status of a hotel booking based on its detailed information:

   - **Class 0: Check-out**
   - **Class 1: Cancel**
   - **Class 2: No-Show**

   This project is the **Classification** task portion of the course data challenge.

   The core modeling strategy is **Stacking**, using a customized L0 layer (LGBM, CatBoost, RandomForest, Keras NN) and an L1 meta-model (LGBM-Meta or KerasNN-Meta) to make the final prediction.

   ## 2. Repository Contents

   - `Booking_code.ipynb`: The main Jupyter Notebook containing all the code. It covers data loading, advanced feature engineering, 10-fold cross-validation (K-Fold) training for the 4 L0 models, training for the 2 L1 meta-models, OOF (Out-of-Fold) evaluation, and the generation of all visualization charts (confusion matrices, feature importance, learning curves).
   - `README.md`: This guide file.

   ## 3. Execution Guide

   ### 3.1. Prerequisites

   This project depends on several Python libraries. It is **strongly recommended** to run this notebook in a **GPU runtime**(like Google Colab), as the CatBoost and Keras NN models in L0, as well as the KerasNN-Meta model in L1, are configured for GPU acceleration.

   You can install all required libraries via `pip`:

   ```
   pip install pandas numpy scikit-learn lightgbm xgboost catboost tensorflow tqdm matplotlib seaborn jupyter
   ```

   ### 3.2. Data

   **Note:** As per the requirements, the data files (`train_data.csv`, `test_data.csv`) are **not** included in this repository.

   1. **Get the Data:** Please download `train_data.csv` and `test_data.csv` from your course platform.

   2. **Place the Data:** The simplest method is to place the `train_data.csv` and `test_data.csv` files in the **same root directory** as `Booking_code.ipynb`. The code will find them automatically.

   3. **Modify Paths:** If your data is stored in a different location, please open `Booking_code.ipynb` and modify the following two lines in **Cell 4** to point to the correct local path:

      ```
      # Cell 4
      train_path = "your/path/to/train_data.csv"
      test_path  = "your/path/to/test_data.csv"
      ```

   ### 3.3. Running the Code

   1. After setting up the environment and data, open `Booking_code.ipynb` in a Jupyter environment (e.g., Google Colab, VS Code, Jupyter Lab).
   2. **Run all cells in order from top to bottom.**
   3. The notebook will execute the complete Stacking pipeline. This will take some time, as the L0 layer trains 3 or 4 models, each with 10 folds (40 models in total).
   4. **Final Outputs:**
      - **Submission Files:** Upon completion, two final submission files will be generated in the root directory: `submission_lgbmcbnn.csv` (from the L1 LGBM-Meta) and `last_soumission.csv` (from the L1 KerasNN-Meta).
      - **Visualizations:** The optional cells at the end of the notebook will generate and save OOF confusion matrices, L0 feature importance plots, and learning curves as `.png` files.


# Prédiction de Popularité Spotify - Blending de Modèles

Ce projet a pour objectif de prédire la popularité de chansons Spotify (tâche de régression) dans le cadre d'un Data Challenge Kaggle.

L'approche principale est un "blending" (ou empilement) de plusieurs modèles de type GBDT (Gradient Boosting Decision Trees) et Random Forest pour améliorer la robustesse et la précision des prédictions.

## Structure du Projet

Le code est entièrement contenu dans le notebook Jupyter `spotify_blending.ipynb`.

Le processus est le suivant :

1. **Chargement et Nettoyage :** Lecture des données d'entraînement (`spotify_train_data.csv`) et de test (`spotify_test_data.csv`).
2. **Feature Engineering :** Création de nouvelles caractéristiques (features) pertinentes, telles que des transformations logarithmiques (pour `duration` et `tempo`), des interactions entre features (ex: `danceability * energy`), et l'encodage cyclique (pour la tonalité `key`).
3. **Entraînement des Modèles de Base (Niveau 1) :** Plusieurs modèles sont entraînés en utilisant une stratégie de validation croisée (K-Fold, 5 plis) pour générer des prédictions "Out-of-Fold" (OOF). Cette méthode permet d'utiliser les prédictions comme nouvelles features pour le modèle de niveau 2 tout en évitant la fuite de données (data leakage).
   - LightGBM (LGBM)
   - XGBoost (XGB) - *Moyenne de plusieurs modèles avec différentes graines aléatoires (bagging).*
   - Random Forest (RF) - *Moyenne de plusieurs graines.*
   - Extra Trees (ET) - *Moyenne de plusieurs graines.*
   - CatBoost
4. **Blending (Modèle de Niveau 2) :** Les prédictions OOF des modèles de base sont utilisées comme features pour entraîner un méta-modèle (un "blender"). Plusieurs types de blenders sont testés :
   - `RidgeCV` (Régression Ridge avec validation croisée) sur 4 modèles (LGBM, XGB, RF, ET).
   - `NNLS` (Non-Negative Least Squares) sur ces 4 mêmes modèles.
   - `RidgeCV` sur un sous-ensemble de 2 modèles (XGB, ET).
5. **Soumission :** Le code génère plusieurs fichiers de soumission (`.csv`) basés sur les différentes stratégies de blending.

## Prérequis

Le code nécessite Python 3 et les bibliothèques suivantes. Vous pouvez les installer via `pip` :

```
pip install pandas numpy scikit-learn lightgbm xgboost category-encoders tqdm catboost matplotlib scipy
```

## Instructions d'Exécution

Suivez ces étapes pour reproduire l'entraînement et générer les fichiers de soumission.

### 1. Placement des données

Ce dépôt ne contient pas les fichiers de données, conformément aux exigences.

1. Veuillez télécharger les fichiers de données sources :
   - `spotify_train_data.csv`
   - `spotify_test_data.csv`
2. Placez ces deux fichiers dans le **même répertoire** que le notebook `spotify_blending.ipynb`.

**Important :** Si vos données se trouvent dans un répertoire différent (par exemple, un dossier nommé `data/`), vous devez **modifier les lignes suivantes** (situées dans la **Cellule 2** du notebook) :

```
# 1) Load and basic cleaning
train = pd.read_csv("spotify_train_data.csv")
test  = pd.read_csv("spotify_test_data.csv")
```

*Remplacez `"spotify_train_data.csv"` par le chemin d'accès complet à votre fichier (ex: `"data/spotify_train_data.csv"`).*

### 2. Exécution du Notebook

1. Ouvrez le notebook `spotify_blending.ipynb` dans un environnement Jupyter (Jupyter Lab, Jupyter Notebook, ou Google Colab).
2. Exécutez toutes les cellules dans l'ordre, de haut en bas ("Run All").

*Note : L'entraînement des modèles (en particulier les modèles multi-graines comme Random Forest et XGBoost, ainsi que CatBoost) peut prendre un temps considérable en fonction de votre machine.*

### 3. Résultats et Soumissions

L'exécution complète du notebook entraînera tous les modèles de base, les modèles de blending, et générera des visualisations d'analyse (distribution de la cible, corrélation, comparaison des modèles).

Les fichiers de soumission suivants seront créés dans le répertoire principal :

- `submission_0911_cat.csv` (Blend RidgeCV de 4 modèles : LGBM, XGB, RF, ET)
- `submission_nnls.csv` (Blend NNLS des 4 mêmes modèles)
- `submission_xgb_et.csv` (Blend RidgeCV de 2 modèles : XGB, ET)

