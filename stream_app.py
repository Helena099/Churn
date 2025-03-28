"""
PRÉDICTION DE CHURN CLIENT - APPLICATION STREAMLIT

Ce script crée une application web interactive pour la prédiction
du churn client, l'analyse des facteurs et la génération de recommandations.

Pour exécuter cette application:
1. Sauvegardez ce code dans un fichier nommé app.py
2. Exécutez la commande: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Configuration de la page
st.set_page_config(page_title="Prédiction du Churn Client", page_icon="📊", layout="wide")

# Titre principal
st.title("💼 Dashboard de Prédiction du Churn Client")
st.markdown("""
Cette application vous permet d'analyser les risques de churn (départ) de vos clients 
et de générer des recommandations pour améliorer la rétention.
""")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page:", 
                       ["Vue d'ensemble", "Analyse des Facteurs", "Segmentation des Clients", 
                        "Prédictions Individuelles", "Recommandations"])

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        # Essayer d'abord de charger les données depuis une URL
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        return df
    except:
        # Si ça ne fonctionne pas, essayer de charger depuis un fichier local
        try:
            df = pd.read_csv("Telco-Customer-Churn.csv")
            return df
        except:
            st.error("Impossible de charger les données. Veuillez télécharger le fichier Telco-Customer-Churn.csv.")
            return None

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        # Charger le modèle
        model_info = joblib.load('models/churn_prediction_model.pkl')
        return model_info
    except:
        st.error("Modèle non trouvé. Veuillez d'abord exécuter le script principal pour entraîner et sauvegarder le modèle.")
        return None

# Fonction pour prétraiter les données
def preprocess_data(df):
    # Copie du dataframe
    df_processed = df.copy()
    
    # Conversion de TotalCharges en numérique
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Imputation des valeurs manquantes
    if df_processed['TotalCharges'].isnull().sum() > 0:
        mask = df_processed['TotalCharges'].isna()
        df_processed.loc[mask, 'TotalCharges'] = df_processed.loc[mask, 'MonthlyCharges'] * df_processed.loc[mask, 'tenure']
    
    # Feature Engineering
    df_processed['tenure_group'] = pd.cut(
        df_processed['tenure'], 
        bins=[0, 12, 24, 36, 48, 60, np.inf],
        labels=['0-1 an', '1-2 ans', '2-3 ans', '3-4 ans', '4-5 ans', '5+ ans']
    )
    
    df_processed['avg_monthly_spend'] = df_processed['TotalCharges'] / df_processed['tenure'].replace(0, 1)
    
    # Fonction pour compter les services
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
    
    df_processed['high_risk_customer'] = ((df_processed['Contract'] == 'Month-to-month') & 
                                         (df_processed['tenure'] < 12)).astype(int)
    
    monthly_threshold = df_processed['MonthlyCharges'].quantile(0.75)
    df_processed['high_charges'] = (df_processed['MonthlyCharges'] > monthly_threshold).astype(int)
    
    df_processed['service_value_ratio'] = df_processed['num_services'] / df_processed['MonthlyCharges']
    
    # Encodage de la variable cible si présente
    if 'Churn' in df_processed.columns:
        df_processed['Churn_binary'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    return df_processed

# Fonction pour prédire le churn
def predict_churn(model_info, customer_data):
    # Récupération du modèle et du préprocesseur
    model = model_info['model']
    preprocessor = model_info['preprocessor']
    
    # Prétraitement des données
    X_processed = preprocessor.transform(customer_data)
    
    # Prédiction des probabilités
    churn_probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Ajout des probabilités au dataframe
    result_df = customer_data.copy()
    result_df['churn_probability'] = churn_probabilities
    
    # Segmentation par niveau de risque
    result_df['risk_segment'] = pd.cut(
        result_df['churn_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Faible Risque', 'Risque Moyen', 'Risque Élevé']
    )
    
    return result_df

# Chargement des données et du modèle
df = load_data()
model_info = load_model()

# Vérification que les données et le modèle sont chargés
if df is None or model_info is None:
    st.warning("Impossible de continuer sans les données et le modèle.")
    st.stop()

# Prétraitement des données
df_processed = preprocess_data(df)

# Page: Vue d'ensemble
if page == "Vue d'ensemble":
    st.header("📈 Vue d'ensemble du Churn Client")
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    
    # Calcul du taux de churn
    churn_rate = df[df['Churn'] == 'Yes'].shape[0] / df.shape[0] * 100
    
    with col1:
        st.metric("Nombre total de clients", df.shape[0])
    
    with col2:
        st.metric("Taux de churn global", f"{churn_rate:.1f}%")
    
    with col3:
        st.metric("Performance du modèle (AUC)", f"{model_info['metrics']['auc']:.4f}")
    
    # Distribution du churn
    st.subheader("Distribution du Churn")
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Statut', 'Nombre de Clients']
    
    fig = px.pie(churn_counts, values='Nombre de Clients', names='Statut', 
                title="Distribution du Churn",
                color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn par type de contrat
    st.subheader("Churn par Type de Contrat")
    contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
    
    fig = px.bar(contract_churn, x='Contract', y='count', color='Churn',
                title="Churn par Type de Contrat",
                color_discrete_sequence=px.colors.qualitative.Safe,
                barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Relation entre ancienneté et charges mensuelles
    st.subheader("Relation entre Ancienneté et Charges Mensuelles")
    fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                    title="Relation entre Ancienneté et Charges Mensuelles",
                    color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig, use_container_width=True)

# Page: Analyse des Facteurs
elif page == "Analyse des Facteurs":
    st.header("🔍 Analyse des Facteurs de Churn")
    
    # Importance des caractéristiques
    st.subheader("Importance des Caractéristiques")
    
    # Récupération des importances de caractéristiques
    feature_importance = model_info.get('feature_importance')
    
    if feature_importance is not None:
        # Affichage des 15 caractéristiques les plus importantes
        top_features = feature_importance.head(15)
        fig = px.bar(top_features, y='Feature', x='Importance', 
                    title="Top 15 des Caractéristiques Importantes",
                    orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Si les importances ne sont pas dans le modèle, les recalculer
        st.warning("Informations d'importance des caractéristiques non disponibles dans le modèle.")
    
    # Analyse des variables numériques
    st.subheader("Analyse des Variables Numériques Clés")
    
    # Sélection des variables numériques
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_var = st.selectbox("Sélectionnez une variable:", numeric_cols)
    
    fig = px.histogram(df, x=selected_var, color='Churn', 
                      marginal='box', 
                      title=f"Distribution de {selected_var} par Statut de Churn",
                      color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des variables catégorielles
    st.subheader("Analyse des Variables Catégorielles")
    
    # Sélection des variables catégorielles principales
    cat_cols = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity']
    selected_cat = st.selectbox("Sélectionnez une variable catégorielle:", cat_cols)
    
    # Calcul des taux de churn par catégorie
    cat_churn_rates = df.groupby([selected_cat])['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    cat_churn_rates.columns = [selected_cat, 'Taux de Churn (%)']
    
    fig = px.bar(cat_churn_rates, x=selected_cat, y='Taux de Churn (%)', 
                title=f"Taux de Churn par {selected_cat}",
                color='Taux de Churn (%)', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# Page: Segmentation des Clients
elif page == "Segmentation des Clients":
    st.header("👥 Segmentation des Clients par Risque")
    
    # Prédiction sur l'ensemble des données
    if 'Churn' in df_processed.columns:
        # Suppression de la colonne cible pour la prédiction
        X = df_processed.drop(['Churn', 'Churn_binary'], axis=1)
    else:
        X = df_processed.copy()
    
    # Exclusion de customerID si présent
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
    
    # Prédiction de churn
    predictions_df = predict_churn(model_info, X)
    
    # Ajout de la colonne churn réelle si disponible
    if 'Churn' in df_processed.columns:
        predictions_df['actual_churn'] = df_processed['Churn_binary'].values
    
    # Distribution des segments de risque
    st.subheader("Distribution des Segments de Risque")
    
    risk_counts = predictions_df['risk_segment'].value_counts().reset_index()
    risk_counts.columns = ['Segment de Risque', 'Nombre de Clients']
    
    fig = px.pie(risk_counts, values='Nombre de Clients', names='Segment de Risque', 
                title="Distribution des Clients par Segment de Risque",
                color_discrete_sequence=['green', 'orange', 'red'])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des caractéristiques par segment
    st.subheader("Caractéristiques par Segment de Risque")
    
    # Séparation des clients par segment
    high_risk = predictions_df[predictions_df['risk_segment'] == 'Risque Élevé']
    medium_risk = predictions_df[predictions_df['risk_segment'] == 'Risque Moyen']
    low_risk = predictions_df[predictions_df['risk_segment'] == 'Faible Risque']
    
    # Sélection des variables clés pour l'analyse
    key_metrics = ['tenure', 'MonthlyCharges', 'num_services']
    selected_metric = st.selectbox("Sélectionnez une métrique:", key_metrics)
    
    # Calcul des moyennes par segment
    segment_stats = predictions_df.groupby('risk_segment')[selected_metric].mean().reset_index()
    
    # Couleurs par segment
    segment_colors = {'Faible Risque': 'green', 'Risque Moyen': 'orange', 'Risque Élevé': 'red'}
    
    fig = px.bar(segment_stats, x='risk_segment', y=selected_metric, 
                title=f"Moyenne de {selected_metric} par Segment de Risque",
                color='risk_segment',
                color_discrete_map=segment_colors)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des contrats par segment
    st.subheader("Types de Contrat par Segment de Risque")
    
    # Calcul de la distribution des contrats par segment
    contract_by_segment = predictions_df.groupby(['risk_segment', 'Contract']).size().reset_index(name='count')
    segment_totals = predictions_df.groupby('risk_segment').size().reset_index(name='total')
    contract_by_segment = contract_by_segment.merge(segment_totals, on='risk_segment')
    contract_by_segment['percentage'] = contract_by_segment['count'] / contract_by_segment['total'] * 100
    
    fig = px.bar(contract_by_segment, x='risk_segment', y='percentage', color='Contract',
                title="Distribution des Types de Contrat par Segment de Risque",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'percentage': 'Pourcentage (%)', 'risk_segment': 'Segment de Risque'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Liste des clients à haut risque
    st.subheader("Top 10 des Clients à Risque Élevé")
    
    # Triage par probabilité de churn décroissante
    high_risk_sorted = high_risk.sort_values('churn_probability', ascending=False)
    
    # Ajout de customerID si disponible
    if 'customerID' in df.columns:
        # Récupérer les customerID correspondants
        high_risk_sorted = high_risk_sorted.reset_index()
        customer_ids = df.loc[high_risk_sorted['index'], 'customerID'].values
        high_risk_sorted['customerID'] = customer_ids
        
        # Sélection des colonnes à afficher
        display_columns = ['customerID', 'churn_probability', 'tenure', 'Contract', 'MonthlyCharges', 'num_services']
    else:
        # Utiliser l'index comme identifiant
        high_risk_sorted = high_risk_sorted.reset_index()
        display_columns = ['index', 'churn_probability', 'tenure', 'Contract', 'MonthlyCharges', 'num_services']
    
    # Affichage des 10 premiers clients à risque élevé
    st.dataframe(high_risk_sorted[display_columns].head(10))

# Page: Prédictions Individuelles
elif page == "Prédictions Individuelles":
    st.header("🔮 Prédictions de Churn Individuelles")
    
    st.markdown("""
    Utilisez ce formulaire pour prédire la probabilité de churn d'un client spécifique.
    Remplissez les champs avec les informations du client pour obtenir une prédiction personnalisée.
    """)
    
    # Interface pour entrer les données du client
    st.subheader("Informations du Client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partenaire", ["No", "Yes"])
        dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
        phone_service = st.selectbox("Service téléphonique", ["No", "Yes"])
        
        # Multiple Lines conditionnel
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Lignes multiples", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
        
        internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
    
    with col2:
        # Services Internet conditionnels
        if internet_service != "No":
            online_security = st.selectbox("Sécurité en ligne", ["No", "Yes"])
            online_backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes"])
            device_protection = st.selectbox("Protection de l'appareil", ["No", "Yes"])
            tech_support = st.selectbox("Support technique", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming films", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Facturation électronique", ["No", "Yes"])
        payment_method = st.selectbox("Méthode de paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    
    with col4:
        monthly_charges = st.slider("Charges mensuelles ($)", 0.0, 150.0, 50.0)
        total_charges = st.number_input("Charges totales ($)", min_value=0.0, value=tenure * monthly_charges)
    
    # Création d'un DataFrame avec les données saisies
    customer_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Bouton pour effectuer la prédiction
    if st.button("Prédire le Churn"):
        # Prétraitement et prédiction
        customer_processed = preprocess_data(customer_data)
        prediction_result = predict_churn(model_info, customer_processed)
        
        # Affichage du résultat
        st.subheader("Résultat de la Prédiction")
        
        # Récupération des résultats
        churn_probability = prediction_result['churn_probability'].values[0]
        risk_segment = prediction_result['risk_segment'].values[0]
        
        # Jauge pour la probabilité de churn
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de Churn (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': churn_probability * 100
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage du segment de risque
        risk_color = {"Faible Risque": "green", "Risque Moyen": "orange", "Risque Élevé": "red"}
        st.markdown(f"""
        <div style="background-color:{risk_color[risk_segment]}; padding:10px; border-radius:5px;">
            <h3 style="color:white; text-align:center;">Segment: {risk_segment}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Facteurs contribuant au churn
        st.subheader("Principaux Facteurs de Risque")
        
        risk_factors = []
        
        # Analyse des facteurs de risque basée sur les valeurs du client
        if contract == "Month-to-month":
            risk_factors.append("Contrat mensuel (plus susceptible au churn)")
        
        if tenure < 12:
            risk_factors.append("Faible ancienneté (client récent)")
        
        if payment_method == "Electronic check":
            risk_factors.append("Paiement par chèque électronique (associé à un taux de churn plus élevé)")
        
        if internet_service == "Fiber optic" and (online_security == "No" or tech_support == "No"):
            risk_factors.append("Fibre optique sans services de sécurité/support")
        
        if monthly_charges > 80:
            risk_factors.append("Charges mensuelles élevées")
        
        # Affichage des facteurs de risque
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("Aucun facteur de risque majeur identifié.")

# Page: Recommandations
elif page == "Recommandations":
    st.header("💡 Recommandations pour Réduire le Churn")
    
    # Sections de recommandations
    st.subheader("Stratégies Générales de Rétention")
    
    general_recs = [
        "Mettre en place un programme de fidélité avec des récompenses progressives basées sur l'ancienneté",
        "Améliorer la qualité du service client et le suivi des plaintes",
        "Organiser des enquêtes de satisfaction régulières pour identifier les problèmes avant qu'ils ne conduisent au churn",
        "Développer une stratégie de communication personnalisée selon le profil de risque du client",
        "Créer des offres spéciales pour les périodes critiques (renouvellement de contrat, premier anniversaire)"
    ]
    
    for rec in general_recs:
        st.markdown(f"- {rec}")
    
    # Recommandations par segment
    st.subheader("Stratégies par Segment de Risque")
    
    tab1, tab2, tab3 = st.tabs(["Clients à Risque Élevé", "Clients à Risque Moyen", "Clients à Faible Risque"])
    
    with tab1:
        high_risk_recs = [
            "Contacter proactivement ces clients pour résoudre leurs problèmes spécifiques",
            "Offrir des promotions exclusives ou réductions pour prolonger leur engagement",
            "Proposer des incitations financières attractives pour convertir les contrats mensuels en contrats à long terme",
            "Mettre en place un suivi personnalisé et régulier par un conseiller dédié",
            "Proposer une évaluation gratuite de leurs besoins actuels et futurs",
            "Offrir des services à valeur ajoutée sans frais supplémentaires pendant une période limitée"
        ]
        
        for rec in high_risk_recs:
            st.markdown(f"- {rec}")
    
    with tab2:
        medium_risk_recs = [
            "Envoyer des enquêtes de satisfaction ciblées pour identifier les points d'amélioration",
            "Mettre en place un programme d'engagement régulier (newsletters, webinaires, événements)",
            "Offrir des options de mise à niveau de service à prix réduit",
            "Proposer des offres de cross-selling adaptées à leur profil d'utilisation",
            "Développer un programme de parrainage avec des avantages pour le client et ses filleuls"
        ]
        
        for rec in medium_risk_recs:
            st.markdown(f"- {rec}")
    
    with tab3:
        low_risk_recs = [
            "Maintenir une relation régulière via des communications personnalisées",
            "Récompenser leur fidélité par des avantages exclusifs",
            "Les solliciter pour des tests de nouveaux produits/services",
            "Développer un programme d'ambassadeurs pour ces clients fidèles",
            "Proposer des offres premium pour augmenter leur valeur à vie (lifetime value)"
        ]
        
        for rec in low_risk_recs:
            st.markdown(f"- {rec}")
    
    # Améliorations des services
    st.subheader("Améliorations des Services et Produits")
    
    service_recs = [
        "Revoir et améliorer les offres et bundles de services pour mieux répondre aux besoins clients",
        "Améliorer la qualité et la fiabilité du service de fibre optique",
        "Renforcer les services de sécurité en ligne et de support technique",
        "Développer de nouvelles fonctionnalités basées sur les retours clients",
        "Optimiser le rapport qualité/prix des services les plus sujets au churn",
        "Établir des normes de qualité de service plus strictes et les communiquer aux clients"
    ]
    
    for rec in service_recs:
        st.markdown(f"- {rec}")
    
    # Optimisation des processus
    st.subheader("Optimisation des Processus")
    
    process_recs = [
        "Simplifier les processus de facturation et de paiement",
        "Améliorer la transparence des contrats et des conditions de service",
        "Faciliter les démarches de renouvellement de contrat",
        "Mettre en place un système d'alerte précoce pour identifier les clients à risque",
        "Optimiser le processus de gestion des réclamations pour résoudre les problèmes plus rapidement"
    ]
    
    for rec in process_recs:
        st.markdown(f"- {rec}")

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("""
### Aide
Cette application utilise un modèle de Machine Learning pour prédire quels clients sont susceptibles de résilier leur abonnement.

Pour utiliser l'application:
1. Explorez les différentes pages via le menu de navigation
2. Utilisez la page "Prédictions Individuelles" pour analyser un client spécifique
3. Consultez les recommandations pour réduire le churn
""")
