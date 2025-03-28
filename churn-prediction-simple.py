"""
PRÉDICTION DE CHURN CLIENT - SCRIPT PRINCIPAL

Ce script présente une version simplifiée du projet complet,
avec toutes les étapes essentielles et des commentaires détaillés.
"""

# ===============================================================
# IMPORTS ET CONFIGURATION
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, roc_auc_score, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os

# Suppression des avertissements pour une meilleure lisibilité
warnings.filterwarnings('ignore')

# Configuration des visualisations
plt.style.use('ggplot')
sns.set(style='whitegrid')


# ===============================================================
# 1. CHARGEMENT DES DONNÉES
# ===============================================================
print("1. CHARGEMENT DES DONNÉES")
print("-" * 40)

# Téléchargement du dataset public "Telco Customer Churn"
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(url)
    print(f"Données chargées avec succès: {df.shape[0]} lignes et {df.shape[1]} colonnes")
except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    # Alternative: charger depuis un fichier local
    df = pd.read_csv("Telco-Customer-Churn.csv")

# Affichage des premières lignes
print("\nAperçu des données:")
print(df.head())

# Vérification des types de données
print("\nTypes de données:")
print(df.dtypes)

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# Distribution de la variable cible
print("\nDistribution du churn:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)


# ===============================================================
# 2. VISUALISATIONS EXPLORATOIRES
# ===============================================================
print("\n2. VISUALISATIONS EXPLORATOIRES")
print("-" * 40)

# Création du répertoire pour les visualisations
if not os.path.exists('visualisations'):
    os.makedirs('visualisations')

# 1. Distribution du churn
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Churn', data=df)
plt.title('Distribution du Churn Client')
plt.xlabel('Churn')
plt.ylabel('Nombre de Clients')

# Ajout des pourcentages sur les barres
total = len(df)
for p in ax.patches:
    percentage = f"{p.get_height() / total * 100:.1f}%"
    ax.annotate(f"{p.get_height()} ({percentage})", 
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='bottom')

plt.savefig('visualisations/churn_distribution.png')
plt.close()

# 2. Churn par type de contrat
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn par Type de Contrat')
plt.xlabel('Type de Contrat')
plt.ylabel('Nombre de Clients')
plt.legend(title='Churn')
plt.savefig('visualisations/churn_by_contract.png')
plt.close()

# 3. Churn par ancienneté
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Churn par Ancienneté du Client')
plt.xlabel('Churn')
plt.ylabel('Ancienneté (mois)')
plt.savefig('visualisations/churn_by_tenure.png')
plt.close()

# 4. Churn par méthode de paiement
plt.figure(figsize=(14, 6))
ax = sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title('Churn par Méthode de Paiement')
plt.xlabel('Méthode de Paiement')
plt.ylabel('Nombre de Clients')
plt.xticks(rotation=15)
plt.legend(title='Churn')
plt.savefig('visualisations/churn_by_payment.png')
plt.close()

print("Visualisations sauvegardées dans le répertoire 'visualisations'")


# ===============================================================
# 3. PRÉPARATION DES DONNÉES
# ===============================================================
print("\n3. PRÉPARATION DES DONNÉES")
print("-" * 40)

# Copie du dataframe original
df_processed = df.copy()

# 1. Gestion des valeurs manquantes
# Conversion de TotalCharges en numérique
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# Vérification des valeurs manquantes
missing_values = df_processed.isnull().sum()
print(f"Valeurs manquantes après conversion: \n{missing_values[missing_values > 0]}")

# Imputation des valeurs manquantes pour TotalCharges
if df_processed['TotalCharges'].isnull().sum() > 0:
    mask = df_processed['TotalCharges'].isna()
    df_processed.loc[mask, 'TotalCharges'] = df_processed.loc[mask, 'MonthlyCharges'] * df_processed.loc[mask, 'tenure']
    print("Valeurs manquantes de TotalCharges imputées.")

# 2. Feature Engineering
print("\nApplication du Feature Engineering...")

# a. Groupe d'ancienneté
df_processed['tenure_group'] = pd.cut(
    df_processed['tenure'], 
    bins=[0, 12, 24, 36, 48, 60, np.inf],
    labels=['0-1 an', '1-2 ans', '2-3 ans', '3-4 ans', '4-5 ans', '5+ ans']
)

# b. Ratio entre charges totales et ancienneté (dépense moyenne mensuelle réelle)
df_processed['avg_monthly_spend'] = df_processed['TotalCharges'] / df_processed['tenure'].replace(0, 1)

# c. Nombre de services souscrits
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Fonction pour compter les services actifs
def count_services(row):
    count = 0
    if row['PhoneService'] == 'Yes':
        count += 1
        if row['MultipleLines'] == 'Yes':
            count += 1
    
    if row['InternetService'] != 'No':
        count += 1
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        for service in services:
            if row[service] == 'Yes':
                count += 1
    
    return count

df_processed['num_services'] = df_processed.apply(count_services, axis=1)

# d. Indicateur de client à risque élevé (contrat mensuel et faible ancienneté)
df_processed['high_risk_customer'] = ((df_processed['Contract'] == 'Month-to-month') & 
                                     (df_processed['tenure'] < 12)).astype(int)

# e. Indicateur de charges élevées
monthly_threshold = df_processed['MonthlyCharges'].quantile(0.75)
df_processed['high_charges'] = (df_processed['MonthlyCharges'] > monthly_threshold).astype(int)

# f. Ratio services/charges
df_processed['service_value_ratio'] = df_processed['num_services'] / df_processed['MonthlyCharges']

# 3. Encodage de la variable cible
df_processed['Churn_binary'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

# 4. Suppression de la colonne customerID
if 'customerID' in df_processed.columns:
    df_processed = df_processed.drop('customerID', axis=1)

# 5. Division en caractéristiques et cible
X = df_processed.drop(['Churn', 'Churn_binary'], axis=1)
y = df_processed['Churn_binary']

# 6. Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Données divisées: {X_train.shape[0]} échantillons d'entraînement, {X_test.shape[0]} échantillons de test")

# 7. Identification des types de colonnes
categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nNombre de caractéristiques catégorielles: {len(categorical_columns)}")
print(f"Nombre de caractéristiques numériques: {len(numerical_columns)}")


# ===============================================================
# 4. CONSTRUCTION DU PIPELINE DE PRÉTRAITEMENT
# ===============================================================
print("\n4. CONSTRUCTION DU PIPELINE DE PRÉTRAITEMENT")
print("-" * 40)

# Préprocesseur pour les variables catégorielles
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Préprocesseur pour les variables numériques
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combinaison des préprocesseurs
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],
    remainder='drop'  # Pour les colonnes qui ne sont pas spécifiées
)

# Application du préprocesseur aux données d'entraînement
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Forme des données d'entraînement après prétraitement: {X_train_processed.shape}")
print(f"Forme des données de test après prétraitement: {X_test_processed.shape}")

# Application de SMOTE pour équilibrer les classes
print("\nApplication de SMOTE pour équilibrer les classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"Forme des données d'entraînement après SMOTE: {X_train_resampled.shape}")
print(f"Distribution des classes après SMOTE: {pd.Series(y_train_resampled).value_counts()}")


# ===============================================================
# 5. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ===============================================================
print("\n5. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("-" * 40)

# Définition des modèles à comparer
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', 
                            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
}

# Dictionnaire pour stocker les résultats
results = {}

# Entraînement et évaluation de chaque modèle
for name, model in models.items():
    print(f"\nEntraînement du modèle: {name}")
    
    # Entraînement du modèle
    model.fit(X_train_resampled, y_train_resampled)
    
    # Prédictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calcul des métriques
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Affichage du rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion:")
    print(conf_matrix)
    
    # Stockage des résultats
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'conf_matrix': conf_matrix
    }
    
    print(f"AUC-ROC: {auc:.4f}")

# Détermination du meilleur modèle
best_model_name = max(results, key=lambda x: results[x]['auc'])
print(f"\nMeilleur modèle selon l'AUC-ROC: {best_model_name} (AUC = {results[best_model_name]['auc']:.4f})")


# ===============================================================
# 6. OPTIMISATION DU MEILLEUR MODÈLE
# ===============================================================
print("\n6. OPTIMISATION DU MEILLEUR MODÈLE")
print("-" * 40)

# Récupération du meilleur modèle
best_model = results[best_model_name]['model']

# Définition de la grille de paramètres selon le modèle
if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'class_weight': ['balanced', None]
    }
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
else:  # XGBoost
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 3, 5]
    }

print(f"Optimisation des hyperparamètres pour {best_model_name}...")
print("Cela peut prendre quelques minutes...")

# Recherche par validation croisée
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Entraînement
grid_search.fit(X_train_resampled, y_train_resampled)

# Meilleurs paramètres
print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score AUC CV: {grid_search.best_score_:.4f}")

# Évaluation sur l'ensemble de test
optimized_model = grid_search.best_estimator_
y_pred_opt = optimized_model.predict(X_test_processed)
y_pred_proba_opt = optimized_model.predict_proba(X_test_processed)[:, 1]

# Calcul des métriques
accuracy_opt = (y_pred_opt == y_test).mean()
auc_opt = roc_auc_score(y_test, y_pred_proba_opt)

print("\nPerformances du modèle optimisé:")
print(f"Accuracy: {accuracy_opt:.4f}")
print(f"AUC-ROC: {auc_opt:.4f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_opt))


# ===============================================================
# 7. ANALYSE DES FACTEURS DE CHURN
# ===============================================================
print("\n7. ANALYSE DES FACTEURS DE CHURN")
print("-" * 40)

# Récupération des importances de caractéristiques selon le type de modèle
if best_model_name == 'Logistic Regression':
    # Pour la régression logistique, on utilise les coefficients
    coef = optimized_model.coef_[0]
    
    # Obtention des noms de caractéristiques après transformation
    feature_names = []
    for name, trans, columns in preprocessor.transformers_:
        if name != 'remainder':
            if name == 'cat':
                # Pour les catégorielles, on récupère les noms après one-hot encoding
                for i, col in enumerate(columns):
                    for category in trans['onehot'].categories_[i]:
                        feature_names.append(f"{col}_{category}")
            else:
                # Pour les numériques, on garde les noms d'origine
                feature_names.extend(columns)
    
    # Création d'un DataFrame pour l'importance des caractéristiques
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coef)
    }).sort_values('Importance', ascending=False)
    
elif best_model_name in ['Random Forest', 'XGBoost']:
    # Pour les modèles basés sur les arbres, on utilise feature_importances_
    importances = optimized_model.feature_importances_
    
    # Obtention des noms de caractéristiques après transformation
    feature_names = []
    for name, trans, columns in preprocessor.transformers_:
        if name != 'remainder':
            if name == 'cat':
                # Pour les catégorielles, on récupère les noms après one-hot encoding
                for i, col in enumerate(columns):
                    for category in trans['onehot'].categories_[i]:
                        feature_names.append(f"{col}_{category}")
            else:
                # Pour les numériques, on garde les noms d'origine
                feature_names.extend(columns)
    
    # Création d'un DataFrame pour l'importance des caractéristiques
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

# Affichage des 15 caractéristiques les plus importantes
print("\nTop 15 des caractéristiques les plus importantes:")
print(feature_importance.head(15))

# Visualisation des caractéristiques importantes
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title(f'Top 15 des Facteurs Influençant le Churn ({best_model_name})')
plt.tight_layout()
plt.savefig('visualisations/feature_importance.png')
plt.close()

print("Visualisation de l'importance des caractéristiques sauvegardée dans 'visualisations/feature_importance.png'")


# ===============================================================
# 8. SEGMENTATION DES CLIENTS À RISQUE
# ===============================================================
print("\n8. SEGMENTATION DES CLIENTS À RISQUE")
print("-" * 40)

# Préparation des données pour la segmentation
X_test_with_proba = X_test.copy()
X_test_with_proba['churn_probability'] = y_pred_proba_opt
X_test_with_proba['actual_churn'] = y_test.values

# Segmentation des clients selon leur risque de churn
X_test_with_proba['risk_segment'] = pd.cut(
    X_test_with_proba['churn_probability'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['Faible Risque', 'Risque Moyen', 'Risque Élevé']
)

# Comptage des clients par segment de risque
risk_counts = X_test_with_proba['risk_segment'].value_counts().sort_index()
print("\nDistribution des clients par segment de risque:")
for segment, count in risk_counts.items():
    percentage = count / len(X_test_with_proba) * 100
    print(f"{segment}: {count} clients ({percentage:.1f}%)")

# Visualisation de la distribution des segments
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='risk_segment', data=X_test_with_proba, palette='viridis')
plt.title('Distribution des Clients par Segment de Risque')
plt.xlabel('Segment de Risque')
plt.ylabel('Nombre de Clients')

# Ajout des pourcentages sur les barres
total = len(X_test_with_proba)
for p in ax.patches:
    percentage = f"{p.get_height() / total * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualisations/risk_segments.png')
plt.close()

# Analyse des taux de churn réels par segment
high_risk = X_test_with_proba[X_test_with_proba['risk_segment'] == 'Risque Élevé']
medium_risk = X_test_with_proba[X_test_with_proba['risk_segment'] == 'Risque Moyen']
low_risk = X_test_with_proba[X_test_with_proba['risk_segment'] == 'Faible Risque']

high_risk_actual = high_risk['actual_churn'].mean() * 100
medium_risk_actual = medium_risk['actual_churn'].mean() * 100
low_risk_actual = low_risk['actual_churn'].mean() * 100

print("\nTaux de churn réel par segment:")
print(f"Risque Élevé: {high_risk_actual:.1f}%")
print(f"Risque Moyen: {medium_risk_actual:.1f}%")
print(f"Risque Faible: {low_risk_actual:.1f}%")

# Visualisation du taux de churn réel par segment
segments = ['Risque Élevé', 'Risque Moyen', 'Risque Faible']
actual_rates = [high_risk_actual, medium_risk_actual, low_risk_actual]

plt.figure(figsize=(10, 6))
bars = plt.bar(segments, actual_rates, color=['red', 'orange', 'green'])
plt.title('Taux de Churn Réel par Segment de Risque')
plt.xlabel('Segment de Risque')
plt.ylabel('Taux de Churn (%)')
plt.ylim(0, 100)

# Ajout des valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{height:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('visualisations/actual_churn_rates.png')
plt.close()

print("Visualisations de segmentation sauvegardées dans le répertoire 'visualisations'")


# ===============================================================
# 9. GÉNÉRATION DE RECOMMANDATIONS BUSINESS
# ===============================================================
print("\n9. GÉNÉRATION DE RECOMMANDATIONS BUSINESS")
print("-" * 40)

# Analyse des caractéristiques des segments
segment_profiles = {}

for feature in ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod']:
    if feature in X_test_with_proba.columns:
        if X_test_with_proba[feature].dtype in ['int64', 'float64']:
            # Variables numériques - calcul de la moyenne
            segment_profiles[feature] = {
                'Risque Élevé': high_risk[feature].mean(),
                'Risque Moyen': medium_risk[feature].mean(),
                'Risque Faible': low_risk[feature].mean()
            }
        else:
            # Variables catégorielles - calcul des proportions
            segment_profiles[feature] = {}
            for segment, data in [('Risque Élevé', high_risk), 
                                ('Risque Moyen', medium_risk), 
                                ('Risque Faible', low_risk)]:
                value_counts = data[feature].value_counts(normalize=True) * 100
                segment_profiles[feature][segment] = value_counts.to_dict()

# Affichage des profils de segment
print("\nProfils des segments:")

for feature, values in segment_profiles.items():
    print(f"\n{feature}:")
    if isinstance(values['Risque Élevé'], dict):
        # Pour les variables catégorielles
        for segment, counts in values.items():
            print(f"  {segment}:")
            for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {cat}: {count:.1f}%")
    else:
        # Pour les variables numériques
        for segment, value in values.items():
            print(f"  {segment}: {value:.2f}")

# Recommandations business basées sur l'analyse
print("\n=== RECOMMANDATIONS BUSINESS ===")

# 1. Recommandations Générales
print("\nRecommandations Générales:")
general_recs = [
    "Mettre en place un programme de fidélité avec des récompenses progressives basées sur l'ancienneté",
    "Améliorer la qualité du service client et le suivi des plaintes",
    "Organiser des enquêtes de satisfaction régulières pour identifier les problèmes avant qu'ils ne conduisent au churn",
    "Développer une stratégie de communication personnalisée selon le profil de risque du client",
    "Créer des offres spéciales pour les périodes critiques (renouvellement de contrat, premier anniversaire)"
]
for i, rec in enumerate(general_recs, 1):
    print(f"  {i}. {rec}")

# 2. Stratégie pour Clients à Risque Élevé
print("\nStratégie pour Clients à Risque Élevé:")
high_risk_recs = [
    "Contacter proactivement ces clients pour résoudre leurs problèmes spécifiques",
    "Offrir des promotions exclusives ou réductions pour prolonger leur engagement",
    "Proposer des incitations financières attractives pour convertir les contrats mensuels en contrats à long terme",
    "Mettre en place un suivi personnalisé et régulier par un conseiller dédié",
    "Proposer une évaluation gratuite de leurs besoins actuels et futurs"
]
for i, rec in enumerate(high_risk_recs, 1):
    print(f"  {i}. {rec}")

# 3. Stratégie pour Clients à Risque Moyen
print("\nStratégie pour Clients à Risque Moyen:")
medium_risk_recs = [
    "Envoyer des enquêtes de satisfaction ciblées pour identifier les points d'amélioration",
    "Mettre en place un programme d'engagement régulier (newsletters, webinaires, événements)",
    "Offrir des options de mise à niveau de service à prix réduit",
    "Proposer des offres de cross-selling adaptées à leur profil d'utilisation",
    "Développer un programme de parrainage avec des avantages pour le client et ses filleuls"
]
for i, rec in enumerate(medium_risk_recs, 1):
    print(f"  {i}. {rec}")

# 4. Améliorations des Services et Produits
print("\nAméliorations des Services et Produits:")
service_recs = [
    "Revoir et améliorer les offres et bundles de services pour mieux répondre aux besoins clients",
    "Améliorer la qualité et la fiabilité du service de fibre optique",
    "Renforcer les services de sécurité en ligne et de support technique",
    "Développer de nouvelles fonctionnalités basées sur les retours clients",
    "Optimiser le rapport qualité/prix des services les plus sujets au churn"
]
for i, rec in enumerate(service_recs, 1):
    print(f"  {i}. {rec}")


# ===============================================================
# 10. SAUVEGARDE DU MODÈLE FINAL
# ===============================================================
print("\n10. SAUVEGARDE DU MODÈLE FINAL")
print("-" * 40)

# Création du répertoire pour les modèles
if not os.path.exists('models'):
    os.makedirs('models')

# Préparation des métadonnées du modèle
model_info = {
    'model': optimized_model,
    'preprocessor': preprocessor,
    'model_name': best_model_name,
    'metrics': {
        'accuracy': accuracy_opt,
        'auc': auc_opt
    },
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance
}

# Sauvegarde du modèle
model_path = 'models/churn_prediction_model.pkl'
joblib.dump(model_info, model_path)
print(f"Modèle sauvegardé avec succès dans: {model_path}")


# ===============================================================
# 11. PRÉDICTION POUR DE NOUVEAUX CLIENTS
# ===============================================================
print("\n11. PRÉDICTION POUR DE NOUVEAUX CLIENTS")
print("-" * 40)

# Exemple de fonction pour prédire le churn de nouveaux clients
def predict_churn_for_new_customers(model_info, new_data):
    """
    Prédit le risque de churn pour de nouveaux clients.
    
    Args:
        model_info: Dictionnaire contenant le modèle et le préprocesseur
        new_data: DataFrame avec les données des nouveaux clients
    
    Returns:
        DataFrame: Données avec les prédictions de churn
    """
    # Récupération du modèle et du préprocesseur
    model = model_info['model']
    preprocessor = model_info['preprocessor']
    
    # Prétraitement des données
    X_processed = preprocessor.transform(new_data)
    
    # Prédiction des probabilités
    churn_probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Ajout des probabilités au dataframe
    result_df = new_data.copy()
    result_df['churn_probability'] = churn_probabilities
    
    # Segmentation par niveau de risque
    result_df['risk_segment'] = pd.cut(
        result_df['churn_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Faible Risque', 'Risque Moyen', 'Risque Élevé']
    )
    
    return result_df

# Création de données de test pour démontrer la prédiction
print("Démonstration de prédiction pour un nouveau client:")

new_customer = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [12],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [70.35],
    'TotalCharges': [845.30]
})

# Application du Feature Engineering au nouveau client
new_customer['tenure_group'] = pd.cut(
    new_customer['tenure'], 
    bins=[0, 12, 24, 36, 48, 60, np.inf],
    labels=['0-1 an', '1-2 ans', '2-3 ans', '3-4 ans', '4-5 ans', '5+ ans']
)

new_customer['avg_monthly_spend'] = new_customer['TotalCharges'] / new_customer['tenure'].replace(0, 1)
new_customer['num_services'] = new_customer.apply(count_services, axis=1)
new_customer['high_risk_customer'] = ((new_customer['Contract'] == 'Month-to-month') & 
                                     (new_customer['tenure'] < 12)).astype(int)
new_customer['high_charges'] = (new_customer['MonthlyCharges'] > monthly_threshold).astype(int)
new_customer['service_value_ratio'] = new_customer['num_services'] / new_customer['MonthlyCharges']

# Prédiction pour le nouveau client
prediction_result = predict_churn_for_new_customers(model_info, new_customer)

# Affichage du résultat
print("\nRésultat de la prédiction:")
print(f"Probabilité de churn: {prediction_result['churn_probability'].values[0]:.2f}")
print(f"Segment de risque: {prediction_result['risk_segment'].values[0]}")

# Recommandations spécifiques basées sur le segment de risque
print("\nRecommandations spécifiques pour ce client:")
risk_segment = prediction_result['risk_segment'].values[0]

if risk_segment == 'Risque Élevé':
    print("1. Contacter immédiatement pour lui proposer une offre de fidélisation")
    print("2. Proposer une promotion sur un contrat à plus long terme")
    print("3. Offrir une remise sur les services de sécurité et support technique")
elif risk_segment == 'Risque Moyen':
    print("1. Inclure dans la prochaine campagne d'emailing avec offres spéciales")
    print("2. Proposer des services complémentaires à prix réduit")
    print("3. Envoyer une enquête de satisfaction pour identifier les points d'amélioration")
else:  # Faible Risque
    print("1. Maintenir la relation client habituelle")
    print("2. Inclure dans le programme de fidélité standard")
    print("3. Proposer des offres de cross-selling pour augmenter la valeur client")


# ===============================================================
# CONCLUSION
# ===============================================================
print("\nCONCLUSION")
print("-" * 40)
print("Le projet de prédiction du churn client a été réalisé avec succès.")
print("Récapitulatif des résultats:")
print(f"- Meilleur modèle: {best_model_name}")
print(f"- AUC sur l'ensemble de test: {auc_opt:.4f}")
print(f"- Accuracy sur l'ensemble de test: {accuracy_opt:.4f}")
print("\nPerspectives d'amélioration:")
print("1. Collecter plus de données comportementales (usage, interactions)")
print("2. Tester des modèles plus complexes (réseaux de neurones)")
print("3. Ajouter des analyses temporelles pour suivre l'évolution du risque")
print("4. Personnaliser les recommandations au niveau individuel")
print("5. Intégrer le modèle dans un système CRM pour automatiser les actions")

print("\nFin du projet.")
